# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 476.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 100 |
| ALERT2 | 98 |
| ALERT2_SKIP | 47 |
| ALERT3 | 260 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 123 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 85
- **Target hits / Stop hits / Partials:** 13 / 111 / 16
- **Avg / median % per leg:** 0.38% / -1.04%
- **Sum % (uncompounded):** 52.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 14 | 29.2% | 7 | 41 | 0 | 0.13% | 6.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.41% | -2.4% |
| BUY @ 3rd Alert (retest2) | 47 | 14 | 29.8% | 7 | 40 | 0 | 0.18% | 8.6% |
| SELL (all) | 92 | 41 | 44.6% | 6 | 70 | 16 | 0.51% | 46.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 92 | 41 | 44.6% | 6 | 70 | 16 | 0.51% | 46.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.41% | -2.4% |
| retest2 (combined) | 139 | 55 | 39.6% | 13 | 110 | 16 | 0.40% | 55.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 368.65 | 377.15 | 378.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 12:15:00 | 366.60 | 372.31 | 375.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 370.90 | 370.75 | 373.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 370.90 | 370.75 | 373.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 370.90 | 370.75 | 373.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 10:30:00 | 368.50 | 371.78 | 372.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 11:15:00 | 369.40 | 371.78 | 372.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 371.40 | 366.46 | 366.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 371.40 | 366.46 | 366.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 15:15:00 | 373.50 | 370.37 | 368.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 13:15:00 | 377.10 | 378.35 | 375.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 14:00:00 | 377.10 | 378.35 | 375.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 379.00 | 378.84 | 376.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 375.40 | 377.46 | 375.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 372.00 | 376.37 | 375.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 371.00 | 376.37 | 375.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 375.35 | 375.15 | 375.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 375.35 | 375.15 | 375.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 378.00 | 375.72 | 375.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 371.85 | 375.72 | 375.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 374.10 | 375.39 | 375.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:15:00 | 367.00 | 375.39 | 375.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 373.20 | 374.96 | 374.99 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 378.45 | 375.32 | 375.09 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 371.00 | 374.34 | 374.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 369.05 | 373.28 | 374.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 373.15 | 373.05 | 373.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 373.15 | 373.05 | 373.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 372.00 | 372.84 | 373.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 368.40 | 372.84 | 373.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 14:15:00 | 375.50 | 371.53 | 372.32 | SL hit (close>static) qty=1.00 sl=374.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 378.00 | 373.79 | 373.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 10:15:00 | 382.00 | 377.14 | 375.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 14:15:00 | 385.05 | 385.37 | 382.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 14:45:00 | 384.05 | 385.37 | 382.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 385.80 | 385.45 | 382.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 369.70 | 385.45 | 382.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 377.20 | 383.80 | 382.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 374.00 | 383.80 | 382.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 362.40 | 379.52 | 380.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 358.00 | 375.22 | 378.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 367.70 | 365.41 | 371.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 367.70 | 365.41 | 371.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 372.20 | 366.77 | 371.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 372.20 | 366.77 | 371.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 367.80 | 366.97 | 371.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 12:45:00 | 365.55 | 366.89 | 370.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 373.60 | 369.11 | 370.66 | SL hit (close>static) qty=1.00 sl=373.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 380.80 | 372.82 | 371.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 382.50 | 374.76 | 372.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 383.10 | 383.42 | 380.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:45:00 | 388.20 | 384.90 | 381.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 378.85 | 383.42 | 381.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 378.85 | 383.42 | 381.90 | SL hit (close<ema400) qty=1.00 sl=381.90 alert=retest1 |

### Cycle 9 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 410.00 | 414.23 | 414.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 408.85 | 413.15 | 413.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 413.00 | 411.03 | 412.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 413.00 | 411.03 | 412.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 413.00 | 411.03 | 412.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 413.00 | 411.03 | 412.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 416.65 | 412.15 | 412.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 419.35 | 412.15 | 412.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 416.55 | 413.34 | 413.27 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 407.80 | 412.91 | 413.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 11:15:00 | 406.40 | 411.60 | 412.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 402.55 | 402.53 | 406.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 11:45:00 | 404.30 | 402.53 | 406.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 405.55 | 403.19 | 405.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 404.10 | 403.19 | 405.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 408.00 | 404.15 | 405.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 408.00 | 404.15 | 405.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 416.70 | 406.66 | 406.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 416.70 | 406.66 | 406.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 422.85 | 409.90 | 408.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 424.10 | 415.53 | 411.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 413.65 | 415.31 | 412.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 11:30:00 | 414.00 | 415.31 | 412.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 412.95 | 414.84 | 412.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 412.95 | 414.84 | 412.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 413.65 | 414.60 | 412.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:45:00 | 414.00 | 414.60 | 412.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 412.65 | 414.21 | 412.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 409.80 | 414.21 | 412.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 414.00 | 414.17 | 412.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 420.00 | 414.17 | 412.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:15:00 | 415.05 | 415.42 | 413.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 410.60 | 414.46 | 413.59 | SL hit (close<static) qty=1.00 sl=411.35 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 412.15 | 413.06 | 413.08 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 420.00 | 414.45 | 413.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 423.30 | 416.22 | 414.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 427.15 | 428.08 | 424.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 10:45:00 | 428.50 | 428.08 | 424.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 430.00 | 428.05 | 424.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:15:00 | 432.75 | 428.05 | 424.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 434.45 | 430.03 | 426.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 419.25 | 431.24 | 430.40 | SL hit (close<static) qty=1.00 sl=423.30 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 417.80 | 428.55 | 429.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 415.55 | 422.04 | 423.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 412.00 | 409.35 | 411.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 412.00 | 409.35 | 411.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 412.00 | 409.35 | 411.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 412.00 | 409.35 | 411.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 415.35 | 410.55 | 412.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 416.00 | 410.55 | 412.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 418.35 | 412.11 | 412.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 422.65 | 412.11 | 412.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 420.10 | 413.71 | 413.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 446.75 | 424.22 | 419.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 445.60 | 446.25 | 439.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 14:45:00 | 443.60 | 446.25 | 439.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 448.25 | 448.62 | 444.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:45:00 | 452.90 | 447.33 | 444.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:45:00 | 450.00 | 446.66 | 445.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-01 12:15:00 | 498.19 | 478.99 | 467.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 466.80 | 494.81 | 497.36 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 502.00 | 491.95 | 491.94 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 487.25 | 491.96 | 492.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 485.00 | 490.56 | 491.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 13:15:00 | 484.05 | 483.18 | 486.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:00:00 | 484.05 | 483.18 | 486.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 475.70 | 481.68 | 485.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 474.80 | 481.68 | 485.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 489.95 | 481.77 | 484.11 | SL hit (close>static) qty=1.00 sl=486.90 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 469.35 | 466.20 | 466.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 477.80 | 468.52 | 467.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 468.70 | 471.54 | 469.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 468.70 | 471.54 | 469.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 468.70 | 471.54 | 469.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 468.70 | 471.54 | 469.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 471.00 | 471.43 | 469.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:30:00 | 473.15 | 472.75 | 470.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:45:00 | 474.00 | 476.62 | 474.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 475.75 | 475.46 | 474.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 474.55 | 475.46 | 474.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 474.70 | 475.31 | 474.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:00:00 | 480.55 | 476.36 | 475.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 15:15:00 | 470.05 | 475.13 | 475.10 | SL hit (close<static) qty=1.00 sl=472.85 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 486.00 | 487.68 | 487.70 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 491.85 | 487.38 | 487.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 494.10 | 488.72 | 487.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 488.65 | 488.71 | 488.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 488.65 | 488.71 | 488.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 488.65 | 488.71 | 488.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 488.65 | 488.71 | 488.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 491.00 | 489.17 | 488.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 485.55 | 488.19 | 487.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 477.70 | 486.09 | 487.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 474.20 | 483.72 | 485.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 479.75 | 474.79 | 479.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 479.75 | 474.79 | 479.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 479.75 | 474.79 | 479.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 479.75 | 474.79 | 479.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 476.80 | 475.19 | 479.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 474.85 | 474.97 | 478.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 15:15:00 | 472.15 | 475.52 | 478.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:45:00 | 475.05 | 473.78 | 476.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 475.00 | 474.14 | 476.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 472.60 | 473.83 | 475.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 480.00 | 476.79 | 476.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 480.00 | 476.79 | 476.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 10:15:00 | 491.85 | 481.38 | 479.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 511.25 | 516.10 | 506.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 10:00:00 | 511.25 | 516.10 | 506.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 504.95 | 513.87 | 506.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 504.95 | 513.87 | 506.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 504.85 | 512.06 | 506.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:30:00 | 505.00 | 512.06 | 506.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 503.90 | 510.43 | 505.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:45:00 | 504.00 | 510.43 | 505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 505.70 | 505.69 | 504.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:30:00 | 506.50 | 505.69 | 504.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 504.05 | 505.36 | 504.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 504.05 | 505.36 | 504.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 500.00 | 504.29 | 504.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 495.60 | 501.87 | 503.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 511.30 | 503.76 | 503.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 511.30 | 503.76 | 503.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 511.30 | 503.76 | 503.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 511.30 | 503.76 | 503.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 507.75 | 504.56 | 504.29 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 502.00 | 504.04 | 504.09 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 516.35 | 506.51 | 505.20 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 502.35 | 505.16 | 505.37 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 12:15:00 | 507.45 | 505.17 | 505.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 13:15:00 | 510.75 | 506.28 | 505.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 525.00 | 526.02 | 520.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 525.00 | 526.02 | 520.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 521.05 | 525.03 | 520.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 521.00 | 525.03 | 520.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 527.80 | 525.58 | 521.18 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 509.90 | 518.48 | 518.96 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 521.65 | 518.20 | 517.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 522.95 | 519.15 | 518.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 13:15:00 | 532.75 | 536.20 | 531.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 13:15:00 | 532.75 | 536.20 | 531.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 532.75 | 536.20 | 531.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 532.75 | 536.20 | 531.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 526.80 | 534.32 | 531.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 527.10 | 534.32 | 531.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 528.20 | 533.10 | 531.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 526.90 | 533.10 | 531.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 523.15 | 529.63 | 529.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 518.85 | 523.82 | 526.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 513.60 | 511.75 | 515.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 513.60 | 511.75 | 515.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 513.60 | 511.75 | 515.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:30:00 | 505.95 | 511.25 | 513.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:00:00 | 504.10 | 505.14 | 508.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:15:00 | 504.15 | 505.36 | 508.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 480.65 | 494.60 | 501.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:15:00 | 478.89 | 492.68 | 499.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:15:00 | 478.94 | 492.68 | 499.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 15:15:00 | 484.00 | 483.92 | 492.71 | SL hit (close>ema200) qty=0.50 sl=483.92 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 503.20 | 494.85 | 494.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 512.20 | 504.04 | 500.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 513.00 | 515.64 | 509.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:30:00 | 514.00 | 515.64 | 509.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 508.60 | 514.24 | 509.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 508.60 | 514.24 | 509.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 512.25 | 513.84 | 509.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:15:00 | 508.45 | 513.84 | 509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 507.65 | 512.60 | 509.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:00:00 | 512.60 | 512.60 | 509.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 516.35 | 514.24 | 511.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 512.00 | 517.34 | 517.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 512.00 | 517.34 | 517.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 509.60 | 515.10 | 516.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 515.10 | 513.67 | 515.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 515.10 | 513.67 | 515.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 515.10 | 513.67 | 515.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 511.00 | 514.55 | 515.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 485.45 | 500.50 | 506.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 485.80 | 485.16 | 492.68 | SL hit (close>ema200) qty=0.50 sl=485.16 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 488.30 | 481.78 | 481.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 12:15:00 | 498.85 | 487.13 | 484.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 489.00 | 489.65 | 486.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:30:00 | 490.10 | 489.65 | 486.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 484.50 | 488.62 | 486.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 484.50 | 488.62 | 486.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 486.50 | 488.20 | 486.43 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 479.85 | 484.56 | 485.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 474.65 | 481.68 | 483.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 480.95 | 476.14 | 479.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 480.95 | 476.14 | 479.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 480.95 | 476.14 | 479.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 480.95 | 476.14 | 479.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 476.10 | 476.13 | 478.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:30:00 | 473.85 | 475.54 | 478.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:00:00 | 473.20 | 475.54 | 478.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:45:00 | 473.00 | 475.34 | 477.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:30:00 | 473.10 | 475.25 | 477.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 473.00 | 473.99 | 476.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 482.00 | 473.99 | 476.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 487.40 | 476.67 | 477.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 487.40 | 476.67 | 477.27 | SL hit (close>static) qty=1.00 sl=486.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 491.40 | 479.62 | 478.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 499.10 | 487.20 | 482.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 539.10 | 541.41 | 528.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:15:00 | 558.00 | 541.41 | 528.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 531.50 | 541.80 | 536.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:30:00 | 558.50 | 543.56 | 537.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 534.00 | 535.23 | 535.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 534.00 | 535.23 | 535.32 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 551.00 | 538.39 | 536.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 555.35 | 541.78 | 538.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 13:15:00 | 539.20 | 541.53 | 538.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 14:00:00 | 539.20 | 541.53 | 538.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 535.65 | 540.35 | 538.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 535.65 | 540.35 | 538.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 537.00 | 539.68 | 538.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 528.70 | 537.87 | 537.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 538.00 | 537.89 | 537.78 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 531.30 | 536.57 | 537.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 12:15:00 | 522.60 | 533.78 | 535.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 530.00 | 527.11 | 531.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 530.00 | 527.11 | 531.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 530.00 | 527.11 | 531.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 530.00 | 527.11 | 531.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 525.15 | 526.72 | 530.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 522.35 | 528.55 | 530.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 535.95 | 530.24 | 530.77 | SL hit (close>static) qty=1.00 sl=534.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 542.50 | 532.69 | 531.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 14:15:00 | 547.00 | 536.96 | 534.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 550.50 | 551.80 | 544.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 550.50 | 551.80 | 544.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 546.25 | 549.75 | 545.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 542.95 | 549.75 | 545.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 545.65 | 548.93 | 545.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 545.05 | 548.93 | 545.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 543.20 | 547.79 | 544.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:00:00 | 543.20 | 547.79 | 544.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 529.50 | 544.13 | 543.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:45:00 | 529.70 | 544.13 | 543.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 528.95 | 541.09 | 542.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 526.40 | 538.15 | 540.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 533.00 | 531.56 | 535.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 533.00 | 531.56 | 535.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 533.00 | 531.56 | 535.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 533.00 | 531.56 | 535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 534.00 | 532.05 | 535.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 534.20 | 532.05 | 535.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 532.95 | 532.23 | 535.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 13:30:00 | 527.00 | 532.02 | 534.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 14:00:00 | 528.35 | 532.02 | 534.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 15:00:00 | 527.80 | 531.17 | 533.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 527.55 | 530.05 | 532.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 532.00 | 529.65 | 531.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:45:00 | 528.35 | 529.65 | 531.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 529.25 | 529.57 | 531.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 15:15:00 | 523.50 | 529.06 | 531.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 522.70 | 524.60 | 526.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 546.00 | 525.59 | 524.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 546.00 | 525.59 | 524.62 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 09:15:00 | 518.55 | 529.53 | 530.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 11:15:00 | 514.90 | 524.76 | 527.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 501.30 | 500.85 | 505.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 11:30:00 | 501.45 | 500.85 | 505.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 504.30 | 501.81 | 503.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:00:00 | 498.75 | 502.08 | 502.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 499.35 | 500.75 | 502.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:00:00 | 498.10 | 500.75 | 502.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 496.45 | 498.43 | 500.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 496.45 | 498.04 | 499.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 498.95 | 498.04 | 499.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 494.05 | 497.24 | 499.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-13 14:15:00 | 504.35 | 500.25 | 500.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 504.35 | 500.25 | 500.11 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 496.20 | 499.40 | 499.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 493.60 | 498.24 | 499.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 499.95 | 498.58 | 499.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 11:15:00 | 499.95 | 498.58 | 499.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 499.95 | 498.58 | 499.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 499.95 | 498.58 | 499.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 504.05 | 499.68 | 499.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 504.05 | 499.68 | 499.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 508.90 | 501.52 | 500.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 15:15:00 | 512.90 | 504.61 | 502.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 570.90 | 577.93 | 565.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:00:00 | 570.90 | 577.93 | 565.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 569.20 | 575.60 | 568.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 568.70 | 575.60 | 568.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 562.40 | 572.96 | 567.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 562.40 | 572.96 | 567.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 566.00 | 571.57 | 567.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 559.80 | 571.57 | 567.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 565.50 | 570.35 | 567.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 566.55 | 570.35 | 567.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 565.10 | 569.30 | 567.11 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 13:15:00 | 560.50 | 565.74 | 565.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 10:15:00 | 558.25 | 562.69 | 564.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 556.50 | 551.12 | 555.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 13:15:00 | 556.50 | 551.12 | 555.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 556.50 | 551.12 | 555.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:45:00 | 561.65 | 551.12 | 555.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 554.75 | 551.85 | 555.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:45:00 | 555.20 | 551.85 | 555.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 550.00 | 551.48 | 554.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 549.50 | 551.48 | 554.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 551.00 | 551.38 | 554.63 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 580.85 | 557.48 | 555.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 585.65 | 577.07 | 571.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 580.65 | 581.69 | 576.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:30:00 | 580.55 | 581.69 | 576.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 585.00 | 582.35 | 577.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 578.00 | 582.35 | 577.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 582.95 | 588.49 | 583.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 582.95 | 588.49 | 583.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 581.75 | 587.14 | 583.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 577.80 | 587.14 | 583.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 577.40 | 585.19 | 582.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:15:00 | 575.00 | 585.19 | 582.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 580.00 | 581.20 | 581.30 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 594.75 | 583.91 | 582.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 600.00 | 592.19 | 587.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 593.55 | 596.47 | 591.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 11:45:00 | 594.00 | 596.47 | 591.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 593.45 | 595.87 | 591.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 598.35 | 596.11 | 592.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 599.00 | 596.11 | 592.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 588.20 | 593.92 | 592.65 | SL hit (close<static) qty=1.00 sl=590.50 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 582.50 | 590.32 | 591.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 569.25 | 583.62 | 587.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 14:15:00 | 571.05 | 569.35 | 575.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 15:00:00 | 571.05 | 569.35 | 575.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 579.85 | 571.07 | 575.10 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 582.00 | 577.33 | 576.89 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 571.55 | 576.18 | 576.40 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 584.60 | 577.86 | 577.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 586.95 | 581.65 | 579.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 12:15:00 | 583.00 | 586.46 | 583.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 12:15:00 | 583.00 | 586.46 | 583.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 583.00 | 586.46 | 583.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:00:00 | 583.00 | 586.46 | 583.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 581.75 | 585.52 | 583.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:30:00 | 582.45 | 585.52 | 583.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 588.10 | 586.03 | 583.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 15:15:00 | 593.00 | 586.03 | 583.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 600.15 | 590.42 | 586.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 593.35 | 590.46 | 586.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:00:00 | 592.30 | 590.83 | 587.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 594.15 | 593.49 | 589.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:15:00 | 602.50 | 593.49 | 589.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 599.75 | 596.49 | 592.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 584.80 | 594.12 | 592.93 | SL hit (close<static) qty=1.00 sl=586.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 587.45 | 591.32 | 591.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 583.90 | 589.84 | 591.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 578.40 | 570.04 | 577.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 578.40 | 570.04 | 577.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 578.40 | 570.04 | 577.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 577.00 | 570.04 | 577.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 575.75 | 571.18 | 577.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 574.80 | 571.18 | 577.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 588.00 | 575.15 | 578.23 | SL hit (close>static) qty=1.00 sl=585.75 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 591.75 | 581.37 | 580.70 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 575.95 | 583.18 | 583.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 574.10 | 581.36 | 582.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 572.60 | 567.27 | 573.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 572.60 | 567.27 | 573.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 572.60 | 567.27 | 573.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 572.60 | 567.27 | 573.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 572.00 | 568.22 | 573.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 572.00 | 568.22 | 573.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 574.35 | 569.44 | 573.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 579.40 | 569.44 | 573.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 575.00 | 570.55 | 573.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 586.00 | 570.55 | 573.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 606.55 | 577.75 | 576.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 610.25 | 584.25 | 579.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 14:15:00 | 767.80 | 768.90 | 745.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:30:00 | 768.05 | 768.90 | 745.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 812.15 | 825.05 | 813.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 814.00 | 825.05 | 813.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 817.65 | 823.57 | 813.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 829.95 | 822.74 | 815.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:15:00 | 822.00 | 822.39 | 815.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 823.95 | 822.16 | 816.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:00:00 | 821.95 | 822.12 | 817.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 810.35 | 820.15 | 817.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 810.35 | 820.15 | 817.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 808.00 | 817.72 | 816.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-10 15:15:00 | 807.85 | 814.05 | 814.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 807.85 | 814.05 | 814.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 801.40 | 811.52 | 813.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 819.90 | 796.94 | 801.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 819.90 | 796.94 | 801.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 819.90 | 796.94 | 801.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 819.90 | 796.94 | 801.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 799.80 | 797.51 | 801.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:30:00 | 798.10 | 797.01 | 800.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 792.10 | 797.29 | 799.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 758.19 | 771.21 | 785.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 752.50 | 771.21 | 785.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 754.25 | 739.51 | 758.12 | SL hit (close>ema200) qty=0.50 sl=739.51 alert=retest2 |

### Cycle 62 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 770.00 | 755.55 | 754.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 787.55 | 766.27 | 759.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 772.90 | 773.65 | 765.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 772.90 | 773.65 | 765.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 772.90 | 773.65 | 765.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 775.40 | 773.65 | 765.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 740.00 | 770.19 | 768.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 740.00 | 770.19 | 768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 748.00 | 765.75 | 766.78 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 788.70 | 763.21 | 760.98 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 755.05 | 775.82 | 776.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 748.85 | 770.43 | 774.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 763.00 | 760.84 | 767.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 766.15 | 760.84 | 767.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 762.05 | 761.08 | 767.00 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 789.10 | 771.76 | 769.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 810.00 | 789.04 | 780.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 801.00 | 803.05 | 792.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 09:15:00 | 797.60 | 803.05 | 792.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 795.10 | 801.46 | 793.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 776.60 | 801.46 | 793.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 796.20 | 800.09 | 793.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:30:00 | 795.85 | 800.09 | 793.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 795.20 | 799.12 | 794.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:00:00 | 795.20 | 799.12 | 794.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 799.45 | 799.18 | 794.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:30:00 | 793.10 | 799.18 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 794.00 | 798.15 | 794.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 794.00 | 798.15 | 794.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 795.00 | 797.52 | 794.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 781.35 | 797.52 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 819.65 | 801.94 | 796.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 12:45:00 | 827.00 | 810.95 | 802.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 819.90 | 832.95 | 825.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 826.65 | 831.67 | 825.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 823.00 | 823.20 | 823.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 823.00 | 823.20 | 823.22 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 835.00 | 824.59 | 823.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 863.95 | 844.61 | 837.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 10:15:00 | 918.45 | 918.80 | 897.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:45:00 | 914.10 | 918.80 | 897.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 909.90 | 915.01 | 900.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:45:00 | 911.75 | 915.01 | 900.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 906.90 | 912.32 | 902.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 906.90 | 912.32 | 902.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 908.00 | 924.72 | 919.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 907.50 | 924.72 | 919.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 906.85 | 921.15 | 917.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 906.05 | 921.15 | 917.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 10:15:00 | 893.30 | 912.51 | 914.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 885.30 | 903.46 | 908.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 871.00 | 865.97 | 878.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 09:30:00 | 869.10 | 865.97 | 878.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 874.65 | 867.71 | 877.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 874.65 | 867.71 | 877.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 867.55 | 867.67 | 876.87 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 897.00 | 880.86 | 880.35 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 840.60 | 876.19 | 879.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 798.60 | 839.66 | 856.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 696.30 | 678.44 | 711.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 696.30 | 678.44 | 711.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 696.30 | 678.44 | 711.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 652.30 | 693.63 | 705.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:00:00 | 677.05 | 666.44 | 673.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 712.00 | 682.01 | 678.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 712.00 | 682.01 | 678.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 743.35 | 700.18 | 687.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 726.10 | 736.76 | 723.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 726.10 | 736.76 | 723.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 726.10 | 736.76 | 723.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 726.10 | 736.76 | 723.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 734.75 | 745.21 | 736.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:00:00 | 734.75 | 745.21 | 736.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 740.05 | 744.17 | 737.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 733.00 | 744.17 | 737.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 751.40 | 744.91 | 739.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:30:00 | 758.70 | 746.55 | 740.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:30:00 | 758.10 | 748.24 | 743.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 757.65 | 746.16 | 744.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 728.40 | 750.76 | 750.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 728.40 | 750.76 | 750.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 714.00 | 743.41 | 747.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 715.00 | 703.44 | 713.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 715.00 | 703.44 | 713.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 715.00 | 703.44 | 713.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:45:00 | 712.30 | 703.44 | 713.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 718.80 | 706.51 | 714.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 723.00 | 706.51 | 714.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 725.30 | 710.27 | 715.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 724.30 | 710.27 | 715.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 712.25 | 710.67 | 715.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 15:15:00 | 709.90 | 711.49 | 714.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 708.55 | 711.05 | 713.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 753.95 | 719.84 | 716.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 753.95 | 719.84 | 716.00 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 715.10 | 726.03 | 726.77 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 732.00 | 727.96 | 727.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 746.75 | 735.56 | 731.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 730.50 | 736.65 | 733.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 730.50 | 736.65 | 733.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 730.50 | 736.65 | 733.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 730.50 | 736.65 | 733.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 723.70 | 734.06 | 732.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 723.70 | 734.06 | 732.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 720.30 | 731.31 | 731.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 717.35 | 728.52 | 730.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 725.90 | 725.09 | 727.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 725.90 | 725.09 | 727.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 725.50 | 725.18 | 727.56 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 748.75 | 729.97 | 729.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 766.70 | 741.77 | 736.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 766.40 | 767.95 | 757.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 766.55 | 767.95 | 757.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 767.00 | 772.02 | 761.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 775.10 | 772.02 | 761.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 817.95 | 826.03 | 817.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 817.95 | 826.03 | 817.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 820.75 | 824.98 | 817.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 819.20 | 824.98 | 817.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 810.90 | 822.16 | 817.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 810.90 | 822.16 | 817.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 810.00 | 819.73 | 816.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 810.00 | 819.73 | 816.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 797.75 | 812.61 | 813.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 796.10 | 809.31 | 812.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 807.00 | 804.76 | 808.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 807.00 | 804.76 | 808.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 807.00 | 804.76 | 808.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 804.85 | 804.76 | 808.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 799.30 | 803.67 | 808.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 799.30 | 803.67 | 808.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 794.95 | 794.20 | 799.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:15:00 | 790.85 | 794.20 | 799.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:00:00 | 789.75 | 793.31 | 798.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 789.00 | 792.93 | 798.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 791.55 | 793.85 | 797.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 799.40 | 794.96 | 797.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 803.00 | 794.96 | 797.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 796.75 | 795.32 | 797.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 802.00 | 799.07 | 798.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 802.00 | 799.07 | 798.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 819.90 | 803.39 | 800.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 810.20 | 811.42 | 806.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:00:00 | 810.20 | 811.42 | 806.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 815.70 | 812.05 | 807.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:00:00 | 819.80 | 813.60 | 808.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 12:15:00 | 901.78 | 892.47 | 876.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 898.40 | 908.45 | 908.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 09:15:00 | 894.00 | 903.93 | 906.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 903.40 | 888.41 | 895.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 903.40 | 888.41 | 895.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 903.40 | 888.41 | 895.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 903.40 | 888.41 | 895.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 901.00 | 890.93 | 895.58 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 908.50 | 899.22 | 898.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 918.90 | 904.51 | 901.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 953.10 | 957.52 | 940.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 953.10 | 957.52 | 940.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 971.50 | 961.89 | 946.67 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 11:15:00 | 938.50 | 946.43 | 946.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 10:15:00 | 926.00 | 937.26 | 941.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 15:15:00 | 872.35 | 864.78 | 875.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 15:15:00 | 872.35 | 864.78 | 875.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 872.35 | 864.78 | 875.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:45:00 | 859.50 | 863.51 | 874.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:45:00 | 859.10 | 862.38 | 872.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 881.05 | 868.51 | 868.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 881.05 | 868.51 | 868.34 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 858.50 | 870.62 | 870.73 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 875.30 | 870.32 | 870.24 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 868.25 | 869.84 | 870.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 858.40 | 865.64 | 867.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 867.70 | 864.55 | 866.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 867.70 | 864.55 | 866.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 867.70 | 864.55 | 866.59 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 877.55 | 868.40 | 868.06 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 857.20 | 866.68 | 867.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 854.15 | 864.18 | 866.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 844.70 | 842.38 | 848.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:45:00 | 843.30 | 842.38 | 848.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 854.00 | 844.70 | 849.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 854.00 | 844.70 | 849.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 861.00 | 847.96 | 850.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 861.00 | 847.96 | 850.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 867.90 | 851.95 | 851.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 868.25 | 855.21 | 853.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 893.30 | 894.08 | 884.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 893.30 | 894.08 | 884.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 881.40 | 891.54 | 884.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 881.40 | 891.54 | 884.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 877.40 | 888.71 | 883.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 876.20 | 888.71 | 883.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 877.70 | 886.51 | 883.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 867.35 | 886.51 | 883.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 911.80 | 913.90 | 906.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 927.50 | 914.03 | 909.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-21 09:15:00 | 1020.25 | 996.90 | 983.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 906.15 | 984.68 | 987.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 864.50 | 929.98 | 958.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 12:15:00 | 793.00 | 790.09 | 813.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 793.00 | 790.09 | 813.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 798.10 | 793.99 | 808.25 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 803.35 | 800.48 | 800.27 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 792.50 | 798.88 | 799.56 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 817.90 | 802.47 | 801.06 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 799.25 | 800.92 | 801.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 11:15:00 | 790.50 | 797.68 | 799.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 791.00 | 788.21 | 792.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 791.00 | 788.21 | 792.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 791.00 | 788.21 | 792.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 791.00 | 788.21 | 792.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 790.00 | 788.57 | 791.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 784.70 | 788.57 | 791.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 771.95 | 785.24 | 790.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 760.90 | 772.03 | 780.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 760.20 | 766.11 | 776.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 798.00 | 766.41 | 770.72 | SL hit (close>static) qty=1.00 sl=796.85 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 786.15 | 775.86 | 774.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 793.10 | 779.31 | 776.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 787.70 | 789.12 | 782.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 787.70 | 789.12 | 782.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 787.70 | 789.12 | 782.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 782.00 | 789.12 | 782.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 793.95 | 790.08 | 783.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:15:00 | 800.00 | 791.12 | 785.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 10:15:00 | 780.00 | 789.79 | 787.12 | SL hit (close<static) qty=1.00 sl=783.40 alert=retest2 |

### Cycle 97 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 771.05 | 784.03 | 784.99 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 788.60 | 785.14 | 784.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 791.25 | 786.89 | 785.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 786.80 | 788.10 | 786.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 11:15:00 | 786.80 | 788.10 | 786.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 786.80 | 788.10 | 786.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 786.80 | 788.10 | 786.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 783.25 | 787.13 | 786.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 783.25 | 787.13 | 786.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 783.95 | 786.50 | 786.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 783.95 | 786.50 | 786.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 791.75 | 787.55 | 786.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 798.00 | 787.55 | 786.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 783.90 | 786.82 | 786.48 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 783.65 | 786.18 | 786.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 779.75 | 783.54 | 784.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 782.80 | 782.65 | 784.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 782.80 | 782.65 | 784.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 782.80 | 782.65 | 784.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 781.10 | 782.65 | 784.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 784.75 | 783.07 | 784.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 784.75 | 783.07 | 784.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 784.50 | 783.36 | 784.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 784.80 | 783.36 | 784.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 780.55 | 782.80 | 783.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:45:00 | 779.05 | 781.62 | 782.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 778.05 | 780.78 | 782.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:00:00 | 777.55 | 779.29 | 780.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 777.50 | 777.89 | 778.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 780.10 | 778.33 | 778.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 772.85 | 777.24 | 778.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 740.10 | 768.89 | 774.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 739.15 | 768.89 | 774.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 738.67 | 768.89 | 774.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 738.62 | 768.89 | 774.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 734.21 | 737.55 | 753.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-28 09:15:00 | 701.14 | 714.78 | 731.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 701.80 | 683.89 | 682.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 704.95 | 690.69 | 685.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 708.00 | 709.68 | 701.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 708.00 | 709.68 | 701.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 725.50 | 732.58 | 725.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 725.50 | 732.58 | 725.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 724.00 | 730.86 | 724.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 724.00 | 730.86 | 724.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 725.95 | 729.88 | 725.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 735.50 | 729.88 | 725.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 718.50 | 729.36 | 728.02 | SL hit (close<static) qty=1.00 sl=723.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 715.00 | 724.93 | 726.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 710.50 | 722.04 | 724.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 695.15 | 691.53 | 701.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:45:00 | 695.50 | 691.53 | 701.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 714.90 | 696.20 | 703.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 714.90 | 696.20 | 703.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 709.40 | 698.84 | 703.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 714.25 | 698.84 | 703.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 716.95 | 707.19 | 706.68 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 702.80 | 709.07 | 709.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 699.00 | 707.06 | 708.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 12:15:00 | 709.20 | 707.48 | 708.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 12:15:00 | 709.20 | 707.48 | 708.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 709.20 | 707.48 | 708.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 710.65 | 707.48 | 708.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 701.00 | 706.19 | 708.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:15:00 | 698.65 | 706.19 | 708.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:45:00 | 698.70 | 704.71 | 707.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 696.85 | 702.31 | 705.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 663.72 | 671.33 | 675.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 663.76 | 671.33 | 675.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 662.01 | 668.56 | 673.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 645.00 | 644.08 | 651.31 | SL hit (close>ema200) qty=0.50 sl=644.08 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 661.65 | 644.23 | 643.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 673.85 | 652.94 | 648.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 669.15 | 675.24 | 667.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 669.15 | 675.24 | 667.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 663.10 | 672.81 | 667.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 663.60 | 672.81 | 667.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 664.10 | 671.07 | 666.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 664.00 | 671.07 | 666.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 663.60 | 669.57 | 666.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:15:00 | 663.00 | 669.57 | 666.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 674.10 | 669.72 | 667.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 679.50 | 670.78 | 667.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 662.45 | 667.39 | 667.28 | SL hit (close<static) qty=1.00 sl=664.85 alert=retest2 |

### Cycle 105 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 662.60 | 666.43 | 666.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 654.45 | 662.66 | 664.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 668.05 | 658.41 | 661.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 668.05 | 658.41 | 661.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 668.05 | 658.41 | 661.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 667.25 | 658.41 | 661.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 668.45 | 660.42 | 661.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 668.45 | 660.42 | 661.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 681.55 | 664.65 | 663.49 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 658.95 | 665.79 | 666.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 655.40 | 661.70 | 663.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 661.90 | 658.95 | 661.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 661.90 | 658.95 | 661.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 661.90 | 658.95 | 661.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 663.95 | 658.95 | 661.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 662.50 | 659.66 | 661.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 662.50 | 659.66 | 661.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 660.70 | 659.87 | 661.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 663.40 | 659.87 | 661.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 661.80 | 660.25 | 661.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 661.80 | 660.25 | 661.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 660.35 | 660.27 | 661.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 661.00 | 660.27 | 661.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 662.25 | 660.67 | 661.33 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 674.50 | 663.33 | 662.42 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 657.00 | 664.72 | 665.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 652.90 | 662.36 | 664.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 14:15:00 | 661.00 | 657.78 | 660.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 14:15:00 | 661.00 | 657.78 | 660.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 661.00 | 657.78 | 660.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 662.70 | 657.78 | 660.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 660.05 | 658.23 | 660.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 662.15 | 658.96 | 660.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 661.90 | 660.03 | 660.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 666.00 | 660.03 | 660.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 657.70 | 659.57 | 660.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 655.80 | 659.57 | 660.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 656.45 | 658.94 | 659.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 655.05 | 657.70 | 659.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 652.00 | 644.21 | 644.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 652.00 | 644.21 | 644.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 652.40 | 647.49 | 645.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 644.60 | 647.31 | 646.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 644.60 | 647.31 | 646.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 644.60 | 647.31 | 646.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 644.60 | 647.31 | 646.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 648.80 | 647.61 | 646.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 649.00 | 647.36 | 646.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 643.05 | 646.50 | 646.29 | SL hit (close<static) qty=1.00 sl=643.80 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 640.30 | 645.26 | 645.75 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 660.00 | 647.50 | 646.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 674.15 | 655.55 | 650.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 609.00 | 659.87 | 659.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 609.00 | 659.87 | 659.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 609.00 | 659.87 | 659.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 606.65 | 659.87 | 659.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 606.65 | 649.23 | 654.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 573.35 | 621.79 | 639.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 569.25 | 567.87 | 591.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 569.25 | 567.87 | 591.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 593.00 | 573.42 | 590.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 606.80 | 573.42 | 590.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 593.50 | 577.44 | 590.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 595.35 | 577.44 | 590.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 587.30 | 581.52 | 590.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 590.70 | 581.52 | 590.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 590.35 | 583.28 | 590.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 590.95 | 583.28 | 590.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 589.40 | 584.51 | 590.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 588.60 | 584.51 | 590.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 588.40 | 586.62 | 590.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 585.25 | 587.53 | 590.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:00:00 | 587.95 | 583.12 | 585.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 583.35 | 582.78 | 584.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 585.50 | 582.78 | 584.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 584.20 | 583.07 | 584.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:45:00 | 584.55 | 583.07 | 584.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 583.50 | 582.61 | 583.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 581.85 | 582.51 | 583.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 581.70 | 582.51 | 583.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:00:00 | 581.05 | 582.22 | 583.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 574.70 | 581.43 | 582.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 576.20 | 575.90 | 578.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 583.40 | 580.04 | 579.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 583.40 | 580.04 | 579.66 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 573.40 | 578.55 | 579.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 571.50 | 575.38 | 577.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 573.35 | 569.64 | 572.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 573.35 | 569.64 | 572.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 573.35 | 569.64 | 572.16 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 579.00 | 574.32 | 573.75 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 566.30 | 573.63 | 573.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 562.60 | 571.43 | 572.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 568.35 | 551.48 | 556.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 568.35 | 551.48 | 556.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 568.35 | 551.48 | 556.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 568.35 | 551.48 | 556.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 565.00 | 554.18 | 557.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 566.15 | 554.18 | 557.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 559.20 | 557.90 | 558.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 560.70 | 557.90 | 558.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 564.00 | 559.75 | 559.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 570.15 | 561.83 | 560.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 574.40 | 577.84 | 574.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 574.40 | 577.84 | 574.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 574.40 | 577.84 | 574.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 571.80 | 577.84 | 574.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 570.60 | 576.39 | 574.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 571.00 | 576.39 | 574.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 572.80 | 575.67 | 574.08 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 566.05 | 571.92 | 572.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 562.00 | 569.69 | 571.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 569.55 | 565.66 | 568.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 569.55 | 565.66 | 568.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 569.55 | 565.66 | 568.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 569.55 | 565.66 | 568.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 571.85 | 566.90 | 568.66 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 572.50 | 569.77 | 569.51 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 568.00 | 569.16 | 569.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 566.45 | 568.61 | 569.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 12:15:00 | 543.50 | 543.12 | 550.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 546.05 | 543.12 | 550.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 544.90 | 542.99 | 548.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 536.80 | 542.99 | 548.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:45:00 | 543.55 | 543.72 | 547.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 542.30 | 546.52 | 547.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 537.00 | 544.77 | 546.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 551.35 | 545.52 | 546.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 551.35 | 545.52 | 546.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 552.00 | 546.82 | 546.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 552.00 | 546.82 | 546.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 554.55 | 548.36 | 547.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 553.35 | 553.58 | 550.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 553.35 | 553.58 | 550.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 553.35 | 553.58 | 550.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 553.35 | 553.58 | 550.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 546.10 | 554.28 | 552.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 546.10 | 554.28 | 552.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 545.45 | 552.51 | 551.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 544.60 | 552.51 | 551.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 549.35 | 551.24 | 551.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 541.95 | 548.30 | 549.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 525.25 | 523.50 | 531.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 525.25 | 523.50 | 531.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 530.30 | 525.26 | 530.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 524.00 | 525.34 | 530.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 545.80 | 531.49 | 530.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 545.80 | 531.49 | 530.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 554.25 | 536.04 | 532.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 549.40 | 551.12 | 544.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:45:00 | 551.00 | 551.12 | 544.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 543.35 | 548.55 | 545.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 543.35 | 548.55 | 545.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 541.60 | 547.16 | 545.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 541.80 | 547.16 | 545.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 539.10 | 543.54 | 544.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 537.50 | 540.86 | 542.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 521.75 | 520.92 | 525.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 521.75 | 520.92 | 525.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 521.75 | 520.92 | 525.02 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 529.50 | 527.08 | 526.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 532.40 | 529.30 | 528.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 531.00 | 531.46 | 529.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:00:00 | 531.00 | 531.46 | 529.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 529.65 | 531.10 | 529.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 529.65 | 531.10 | 529.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 531.20 | 531.12 | 529.77 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 523.00 | 528.09 | 528.63 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 531.15 | 527.97 | 527.70 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 525.25 | 527.59 | 527.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 521.00 | 526.05 | 526.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 504.25 | 502.76 | 508.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 504.25 | 502.76 | 508.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 509.05 | 504.46 | 507.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 510.00 | 504.46 | 507.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 511.35 | 505.84 | 508.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 512.10 | 505.84 | 508.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 514.35 | 508.03 | 508.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 514.35 | 508.03 | 508.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 508.50 | 508.13 | 508.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:00:00 | 507.20 | 507.94 | 508.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 481.84 | 486.51 | 492.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 456.48 | 464.89 | 474.47 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 130 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 425.75 | 413.54 | 413.13 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 413.00 | 415.95 | 416.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 407.05 | 413.68 | 414.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 15:15:00 | 408.00 | 407.95 | 410.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 401.30 | 407.95 | 410.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 398.95 | 406.15 | 409.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 397.00 | 404.60 | 408.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 437.25 | 412.46 | 410.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 437.25 | 412.46 | 410.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 444.00 | 434.79 | 424.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 437.70 | 440.96 | 433.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 435.80 | 440.96 | 433.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 441.50 | 441.84 | 437.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 440.00 | 441.84 | 437.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 441.70 | 442.52 | 440.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 441.50 | 442.52 | 440.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 440.10 | 442.04 | 440.10 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 397.30 | 430.84 | 435.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 358.60 | 406.73 | 422.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 371.50 | 370.84 | 392.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 380.00 | 370.84 | 392.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 375.40 | 362.85 | 369.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 375.40 | 362.85 | 369.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 368.45 | 363.97 | 369.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 367.00 | 364.58 | 369.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 367.35 | 360.74 | 363.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 367.25 | 362.04 | 363.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 391.40 | 367.91 | 366.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 391.40 | 367.91 | 366.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 405.65 | 386.55 | 377.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 398.40 | 400.49 | 390.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:45:00 | 398.15 | 400.49 | 390.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 400.50 | 399.69 | 394.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 413.30 | 399.69 | 394.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 404.55 | 401.42 | 395.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 404.90 | 401.18 | 398.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 387.50 | 396.53 | 397.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 387.50 | 396.53 | 397.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 384.75 | 394.17 | 396.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 378.90 | 376.14 | 381.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 378.90 | 376.14 | 381.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 385.00 | 377.91 | 382.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 385.00 | 377.91 | 382.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 387.00 | 379.73 | 382.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 387.00 | 379.73 | 382.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 381.95 | 382.93 | 383.53 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 385.85 | 384.12 | 383.96 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 364.55 | 380.53 | 382.40 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 380.70 | 378.16 | 377.90 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 376.50 | 377.71 | 377.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 371.95 | 376.56 | 377.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 377.35 | 375.50 | 376.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 377.35 | 375.50 | 376.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 377.35 | 375.50 | 376.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 377.35 | 375.50 | 376.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 375.70 | 375.54 | 376.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:45:00 | 376.45 | 375.54 | 376.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 377.40 | 375.91 | 376.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:00:00 | 377.40 | 375.91 | 376.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 378.20 | 376.37 | 376.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 369.00 | 376.37 | 376.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 350.55 | 358.06 | 365.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 357.25 | 353.77 | 360.41 | SL hit (close>ema200) qty=0.50 sl=353.77 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 371.00 | 360.61 | 360.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 373.95 | 363.28 | 361.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 372.00 | 373.32 | 367.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 370.30 | 373.32 | 367.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 368.20 | 372.30 | 367.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 368.20 | 372.30 | 367.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 368.15 | 371.47 | 367.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 368.15 | 371.47 | 367.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 370.50 | 371.28 | 368.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:15:00 | 366.00 | 371.28 | 368.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 367.10 | 370.44 | 368.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 372.50 | 368.50 | 367.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 366.00 | 367.05 | 367.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 366.00 | 367.05 | 367.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 362.55 | 366.15 | 366.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 350.35 | 347.25 | 353.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 351.40 | 347.25 | 353.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 356.40 | 349.62 | 352.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 356.70 | 349.62 | 352.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 360.05 | 351.71 | 352.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 360.70 | 351.71 | 352.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 357.30 | 353.74 | 353.62 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 343.35 | 351.61 | 352.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 338.50 | 348.99 | 351.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 350.55 | 336.10 | 340.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 350.55 | 336.10 | 340.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 350.55 | 336.10 | 340.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 350.55 | 336.10 | 340.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 347.95 | 338.47 | 340.95 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 352.60 | 343.75 | 343.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 357.30 | 350.39 | 347.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 357.95 | 360.52 | 357.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 357.95 | 360.52 | 357.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 357.95 | 360.52 | 357.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:30:00 | 358.60 | 360.52 | 357.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 359.00 | 360.22 | 357.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 370.00 | 360.22 | 357.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-13 14:15:00 | 407.00 | 400.96 | 393.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 413.75 | 416.86 | 417.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 410.25 | 415.54 | 416.47 | Break + close below crossover candle low |

### Cycle 146 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 443.65 | 414.07 | 413.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 446.15 | 439.65 | 432.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 442.65 | 443.11 | 437.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 442.65 | 443.11 | 437.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 441.00 | 442.88 | 438.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 439.00 | 442.88 | 438.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 444.85 | 443.27 | 439.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 446.40 | 443.27 | 439.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 463.70 | 444.58 | 441.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 14:15:00 | 491.04 | 482.75 | 470.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-05-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 11:15:00 | 481.35 | 487.40 | 487.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 13:15:00 | 478.65 | 484.41 | 486.04 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 10:30:00 | 368.50 | 2024-05-21 09:15:00 | 371.40 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-05-15 11:15:00 | 369.40 | 2024-05-21 09:15:00 | 371.40 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-05-29 09:15:00 | 368.40 | 2024-05-29 14:15:00 | 375.50 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-05-30 09:15:00 | 371.00 | 2024-05-30 10:15:00 | 378.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-06-05 12:45:00 | 365.55 | 2024-06-06 09:15:00 | 373.60 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest1 | 2024-06-11 09:45:00 | 388.20 | 2024-06-11 14:15:00 | 378.85 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-06-12 09:15:00 | 382.00 | 2024-06-20 09:15:00 | 420.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 09:15:00 | 420.00 | 2024-07-03 13:15:00 | 410.60 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-07-03 13:15:00 | 415.05 | 2024-07-03 13:15:00 | 410.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-07-08 13:15:00 | 432.75 | 2024-07-10 10:15:00 | 419.25 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-07-09 09:15:00 | 434.45 | 2024-07-10 10:15:00 | 419.25 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-07-29 09:45:00 | 452.90 | 2024-08-01 12:15:00 | 498.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 09:45:00 | 450.00 | 2024-08-01 12:15:00 | 495.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-09 15:15:00 | 474.80 | 2024-08-12 11:15:00 | 489.95 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-08-13 10:00:00 | 472.80 | 2024-08-19 15:15:00 | 469.35 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-08-13 11:30:00 | 474.00 | 2024-08-19 15:15:00 | 469.35 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2024-08-21 09:30:00 | 473.15 | 2024-08-23 15:15:00 | 470.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-08-22 13:45:00 | 474.00 | 2024-08-29 14:15:00 | 486.00 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2024-08-23 09:45:00 | 475.75 | 2024-08-29 14:15:00 | 486.00 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2024-08-23 10:15:00 | 474.55 | 2024-08-29 14:15:00 | 486.00 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2024-08-23 12:00:00 | 480.55 | 2024-08-29 14:15:00 | 486.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2024-08-26 09:15:00 | 479.35 | 2024-08-29 14:15:00 | 486.00 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2024-09-03 11:30:00 | 474.85 | 2024-09-05 11:15:00 | 480.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-03 15:15:00 | 472.15 | 2024-09-05 11:15:00 | 480.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-09-04 12:45:00 | 475.05 | 2024-09-05 11:15:00 | 480.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-04 14:15:00 | 475.00 | 2024-09-05 11:15:00 | 480.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-10-03 11:30:00 | 505.95 | 2024-10-07 10:15:00 | 480.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 11:00:00 | 504.10 | 2024-10-07 11:15:00 | 478.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:15:00 | 504.15 | 2024-10-07 11:15:00 | 478.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:30:00 | 505.95 | 2024-10-07 15:15:00 | 484.00 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2024-10-04 11:00:00 | 504.10 | 2024-10-07 15:15:00 | 484.00 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2024-10-04 12:15:00 | 504.15 | 2024-10-07 15:15:00 | 484.00 | STOP_HIT | 0.50 | 4.00% |
| BUY | retest2 | 2024-10-14 14:00:00 | 512.60 | 2024-10-17 14:15:00 | 512.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-10-15 10:00:00 | 516.35 | 2024-10-17 14:15:00 | 512.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-10-21 09:30:00 | 511.00 | 2024-10-22 10:15:00 | 485.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 511.00 | 2024-10-23 12:15:00 | 485.80 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2024-11-04 09:30:00 | 473.85 | 2024-11-05 09:15:00 | 487.40 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-11-04 10:00:00 | 473.20 | 2024-11-05 09:15:00 | 487.40 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-11-04 10:45:00 | 473.00 | 2024-11-05 09:15:00 | 487.40 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-11-04 12:30:00 | 473.10 | 2024-11-05 09:15:00 | 487.40 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-11-11 10:30:00 | 558.50 | 2024-11-12 09:15:00 | 534.00 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2024-11-18 09:30:00 | 522.35 | 2024-11-18 11:15:00 | 535.95 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-11-25 13:30:00 | 527.00 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-11-25 14:00:00 | 528.35 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2024-11-25 15:00:00 | 527.80 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-11-26 10:15:00 | 527.55 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-11-26 15:15:00 | 523.50 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2024-11-28 10:30:00 | 522.70 | 2024-12-02 09:15:00 | 546.00 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2024-12-11 15:00:00 | 498.75 | 2024-12-13 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-12 09:30:00 | 499.35 | 2024-12-13 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-12 10:00:00 | 498.10 | 2024-12-13 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-12-12 15:15:00 | 496.45 | 2024-12-13 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-08 14:15:00 | 598.35 | 2025-01-09 11:15:00 | 588.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-01-09 09:15:00 | 599.00 | 2025-01-09 11:15:00 | 588.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-01-16 15:15:00 | 593.00 | 2025-01-21 10:15:00 | 584.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-01-17 09:45:00 | 600.15 | 2025-01-21 10:15:00 | 584.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-01-17 11:15:00 | 593.35 | 2025-01-21 13:15:00 | 587.45 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-17 12:00:00 | 592.30 | 2025-01-21 13:15:00 | 587.45 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-20 10:15:00 | 602.50 | 2025-01-21 13:15:00 | 587.45 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-01-20 12:30:00 | 599.75 | 2025-01-21 13:15:00 | 587.45 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-01-23 11:15:00 | 574.80 | 2025-01-23 12:15:00 | 588.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-02-07 12:30:00 | 829.95 | 2025-02-10 15:15:00 | 807.85 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-02-07 14:15:00 | 822.00 | 2025-02-10 15:15:00 | 807.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-02-10 09:15:00 | 823.95 | 2025-02-10 15:15:00 | 807.85 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-02-10 10:00:00 | 821.95 | 2025-02-10 15:15:00 | 807.85 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-02-12 13:30:00 | 798.10 | 2025-02-14 09:15:00 | 758.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 792.10 | 2025-02-14 09:15:00 | 752.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 798.10 | 2025-02-17 09:15:00 | 754.25 | STOP_HIT | 0.50 | 5.49% |
| SELL | retest2 | 2025-02-13 11:30:00 | 792.10 | 2025-02-17 09:15:00 | 754.25 | STOP_HIT | 0.50 | 4.78% |
| BUY | retest2 | 2025-03-07 12:45:00 | 827.00 | 2025-03-12 09:15:00 | 823.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-03-11 09:45:00 | 819.90 | 2025-03-12 09:15:00 | 823.00 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-03-11 10:45:00 | 826.65 | 2025-03-12 09:15:00 | 823.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-04-09 09:15:00 | 652.30 | 2025-04-15 09:15:00 | 712.00 | STOP_HIT | 1.00 | -9.15% |
| SELL | retest2 | 2025-04-11 13:00:00 | 677.05 | 2025-04-15 09:15:00 | 712.00 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-04-22 10:30:00 | 758.70 | 2025-04-25 10:15:00 | 728.40 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-04-22 14:30:00 | 758.10 | 2025-04-25 10:15:00 | 728.40 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-04-24 10:15:00 | 757.65 | 2025-04-25 10:15:00 | 728.40 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-04-30 15:15:00 | 709.90 | 2025-05-05 09:15:00 | 753.95 | STOP_HIT | 1.00 | -6.21% |
| SELL | retest2 | 2025-05-02 10:15:00 | 708.55 | 2025-05-05 09:15:00 | 753.95 | STOP_HIT | 1.00 | -6.41% |
| SELL | retest2 | 2025-05-22 12:15:00 | 790.85 | 2025-05-23 14:15:00 | 802.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-22 13:00:00 | 789.75 | 2025-05-23 14:15:00 | 802.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-05-22 13:30:00 | 789.00 | 2025-05-23 14:15:00 | 802.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-05-23 09:15:00 | 791.55 | 2025-05-23 14:15:00 | 802.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-05-27 11:00:00 | 819.80 | 2025-05-30 12:15:00 | 901.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:45:00 | 859.50 | 2025-06-25 09:15:00 | 881.05 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-06-23 10:45:00 | 859.10 | 2025-06-25 09:15:00 | 881.05 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-07-14 10:00:00 | 927.50 | 2025-07-21 09:15:00 | 1020.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-06 15:00:00 | 760.90 | 2025-08-07 15:15:00 | 798.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-08-07 09:30:00 | 760.20 | 2025-08-07 15:15:00 | 798.00 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2025-08-11 14:15:00 | 800.00 | 2025-08-12 10:15:00 | 780.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-08-20 13:45:00 | 779.05 | 2025-08-25 09:15:00 | 740.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 14:45:00 | 778.05 | 2025-08-25 09:15:00 | 739.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:00:00 | 777.55 | 2025-08-25 09:15:00 | 738.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:45:00 | 777.50 | 2025-08-25 09:15:00 | 738.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 15:00:00 | 772.85 | 2025-08-26 09:15:00 | 734.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 13:45:00 | 779.05 | 2025-08-28 09:15:00 | 701.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-20 14:45:00 | 778.05 | 2025-08-28 09:15:00 | 700.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 13:00:00 | 777.55 | 2025-08-28 09:15:00 | 699.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 12:45:00 | 777.50 | 2025-08-28 09:15:00 | 699.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 15:00:00 | 772.85 | 2025-08-28 09:15:00 | 695.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-09 09:15:00 | 735.50 | 2025-09-10 09:15:00 | 718.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-09-17 14:15:00 | 698.65 | 2025-09-25 09:15:00 | 663.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:45:00 | 698.70 | 2025-09-25 09:15:00 | 663.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 696.85 | 2025-09-25 11:15:00 | 662.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:15:00 | 698.65 | 2025-09-29 15:15:00 | 645.00 | STOP_HIT | 0.50 | 7.68% |
| SELL | retest2 | 2025-09-17 14:45:00 | 698.70 | 2025-09-29 15:15:00 | 645.00 | STOP_HIT | 0.50 | 7.69% |
| SELL | retest2 | 2025-09-18 10:00:00 | 696.85 | 2025-09-29 15:15:00 | 645.00 | STOP_HIT | 0.50 | 7.44% |
| BUY | retest2 | 2025-10-08 09:15:00 | 679.50 | 2025-10-08 13:15:00 | 662.45 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-10-23 11:15:00 | 655.80 | 2025-10-29 11:15:00 | 652.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-10-23 12:00:00 | 656.45 | 2025-10-29 11:15:00 | 652.00 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-10-23 13:45:00 | 655.05 | 2025-10-29 11:15:00 | 652.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-10-31 09:15:00 | 649.00 | 2025-10-31 09:15:00 | 643.05 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-10 13:15:00 | 588.60 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-11-10 14:30:00 | 588.40 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2025-11-11 09:15:00 | 585.25 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-12 11:00:00 | 587.95 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-11-13 10:45:00 | 581.85 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-13 11:15:00 | 581.70 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-13 12:00:00 | 581.05 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-14 09:15:00 | 574.70 | 2025-11-17 14:15:00 | 583.40 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-09 09:15:00 | 536.80 | 2025-12-11 12:15:00 | 552.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-12-09 12:45:00 | 543.55 | 2025-12-11 12:15:00 | 552.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-12-10 14:00:00 | 542.30 | 2025-12-11 12:15:00 | 552.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-10 14:30:00 | 537.00 | 2025-12-11 12:15:00 | 552.00 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-12-19 11:15:00 | 524.00 | 2025-12-22 09:15:00 | 545.80 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2026-01-13 13:00:00 | 507.20 | 2026-01-19 09:15:00 | 481.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:00:00 | 507.20 | 2026-01-21 09:15:00 | 456.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 10:30:00 | 397.00 | 2026-02-09 09:15:00 | 437.25 | STOP_HIT | 1.00 | -10.14% |
| SELL | retest2 | 2026-02-19 12:00:00 | 367.00 | 2026-02-23 11:15:00 | 391.40 | STOP_HIT | 1.00 | -6.65% |
| SELL | retest2 | 2026-02-23 09:45:00 | 367.35 | 2026-02-23 11:15:00 | 391.40 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2026-02-23 11:00:00 | 367.25 | 2026-02-23 11:15:00 | 391.40 | STOP_HIT | 1.00 | -6.58% |
| BUY | retest2 | 2026-02-26 09:15:00 | 413.30 | 2026-03-02 11:15:00 | 387.50 | STOP_HIT | 1.00 | -6.24% |
| BUY | retest2 | 2026-02-26 11:15:00 | 404.55 | 2026-03-02 11:15:00 | 387.50 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2026-02-27 15:00:00 | 404.90 | 2026-03-02 11:15:00 | 387.50 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2026-03-13 09:15:00 | 369.00 | 2026-03-16 10:15:00 | 350.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 369.00 | 2026-03-16 14:15:00 | 357.25 | STOP_HIT | 0.50 | 3.18% |
| BUY | retest2 | 2026-03-20 09:15:00 | 372.50 | 2026-03-20 12:15:00 | 366.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-08 09:15:00 | 370.00 | 2026-04-13 14:15:00 | 407.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:15:00 | 446.40 | 2026-05-05 14:15:00 | 491.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 463.70 | 2026-05-11 11:15:00 | 481.35 | STOP_HIT | 1.00 | 3.81% |
