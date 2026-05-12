# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 670.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 257 |
| ALERT1 | 157 |
| ALERT2 | 154 |
| ALERT2_SKIP | 95 |
| ALERT3 | 385 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 167 |
| PARTIAL | 20 |
| TARGET_HIT | 4 |
| STOP_HIT | 171 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 195 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 124
- **Target hits / Stop hits / Partials:** 4 / 171 / 20
- **Avg / median % per leg:** 0.29% / -0.50%
- **Sum % (uncompounded):** 57.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 82 | 15 | 18.3% | 2 | 80 | 0 | -0.79% | -65.1% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.13% | -2.1% |
| BUY @ 3rd Alert (retest2) | 81 | 15 | 18.5% | 2 | 79 | 0 | -0.78% | -63.0% |
| SELL (all) | 113 | 56 | 49.6% | 2 | 91 | 20 | 1.08% | 122.2% |
| SELL @ 2nd Alert (retest1) | 11 | 9 | 81.8% | 0 | 7 | 4 | 2.83% | 31.1% |
| SELL @ 3rd Alert (retest2) | 102 | 47 | 46.1% | 2 | 84 | 16 | 0.89% | 91.1% |
| retest1 (combined) | 12 | 9 | 75.0% | 0 | 8 | 4 | 2.42% | 29.0% |
| retest2 (combined) | 183 | 62 | 33.9% | 4 | 163 | 16 | 0.15% | 28.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 369.80 | 377.46 | 378.08 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 12:15:00 | 384.25 | 374.08 | 373.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 387.25 | 378.24 | 375.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 14:15:00 | 381.20 | 382.14 | 379.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 14:45:00 | 380.10 | 382.14 | 379.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 379.60 | 381.41 | 379.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 09:30:00 | 378.85 | 381.41 | 379.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 379.15 | 380.96 | 379.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:30:00 | 378.20 | 380.96 | 379.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 382.55 | 381.27 | 379.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 13:15:00 | 383.40 | 381.07 | 379.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 13:45:00 | 382.95 | 381.48 | 380.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 14:30:00 | 384.30 | 381.95 | 380.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-23 09:15:00 | 387.20 | 384.46 | 382.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 385.80 | 384.73 | 383.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 09:30:00 | 391.30 | 386.57 | 384.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 11:00:00 | 390.05 | 387.26 | 385.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 14:15:00 | 382.85 | 385.60 | 385.18 | SL hit (close<static) qty=1.00 sl=383.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 381.40 | 384.46 | 384.88 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 390.90 | 385.36 | 385.19 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 11:15:00 | 382.30 | 385.05 | 385.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 12:15:00 | 380.70 | 384.18 | 384.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 11:15:00 | 385.00 | 381.61 | 382.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 11:15:00 | 385.00 | 381.61 | 382.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 11:15:00 | 385.00 | 381.61 | 382.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 11:30:00 | 388.15 | 381.61 | 382.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 380.25 | 381.34 | 382.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 14:15:00 | 378.20 | 380.87 | 382.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 09:30:00 | 378.50 | 380.29 | 381.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 10:45:00 | 378.50 | 380.13 | 381.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 12:45:00 | 378.50 | 379.60 | 380.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 380.00 | 379.05 | 380.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:30:00 | 382.00 | 379.05 | 380.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 376.80 | 378.60 | 379.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 12:15:00 | 374.25 | 378.43 | 379.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 15:15:00 | 383.00 | 378.68 | 379.28 | SL hit (close>static) qty=1.00 sl=380.25 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 384.95 | 380.50 | 380.04 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 10:15:00 | 379.00 | 380.76 | 380.94 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 14:15:00 | 382.40 | 381.07 | 381.00 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 379.75 | 380.82 | 380.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 12:15:00 | 377.55 | 380.03 | 380.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 379.95 | 379.66 | 380.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 14:15:00 | 379.95 | 379.66 | 380.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 379.95 | 379.66 | 380.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 379.95 | 379.66 | 380.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 379.35 | 379.60 | 380.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 378.50 | 379.60 | 380.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 379.50 | 379.58 | 380.12 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 10:15:00 | 382.45 | 380.50 | 380.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 12:15:00 | 386.70 | 381.93 | 381.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 409.15 | 411.88 | 405.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 10:00:00 | 409.15 | 411.88 | 405.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 410.50 | 413.14 | 409.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:00:00 | 410.50 | 413.14 | 409.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 412.00 | 412.91 | 410.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 12:15:00 | 413.00 | 412.91 | 410.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 13:30:00 | 413.10 | 412.94 | 410.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 14:45:00 | 413.00 | 413.10 | 410.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 11:30:00 | 413.20 | 412.94 | 411.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 412.90 | 412.93 | 411.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 12:45:00 | 412.25 | 412.93 | 411.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 13:15:00 | 412.45 | 412.84 | 411.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 13:30:00 | 412.15 | 412.84 | 411.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 410.90 | 412.45 | 411.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:45:00 | 410.10 | 412.45 | 411.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 407.00 | 411.36 | 411.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-16 15:15:00 | 407.00 | 411.36 | 411.24 | SL hit (close<static) qty=1.00 sl=409.20 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 410.45 | 413.12 | 413.36 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 13:15:00 | 419.40 | 414.29 | 413.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 10:15:00 | 425.15 | 419.11 | 416.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 11:15:00 | 421.75 | 423.51 | 420.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-26 12:00:00 | 421.75 | 423.51 | 420.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 422.60 | 423.33 | 421.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 14:00:00 | 423.35 | 423.33 | 421.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 14:15:00 | 418.70 | 422.40 | 421.00 | SL hit (close<static) qty=1.00 sl=420.10 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 10:15:00 | 416.30 | 420.04 | 420.16 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 434.50 | 421.64 | 420.73 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 423.70 | 424.26 | 424.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 14:15:00 | 417.60 | 421.83 | 423.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 11:15:00 | 410.55 | 410.28 | 412.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 11:30:00 | 410.05 | 410.28 | 412.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 412.25 | 410.23 | 411.77 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 13:15:00 | 415.60 | 413.02 | 412.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 416.75 | 413.77 | 413.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 15:15:00 | 412.80 | 413.57 | 413.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 15:15:00 | 412.80 | 413.57 | 413.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 412.80 | 413.57 | 413.07 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 411.00 | 412.94 | 412.96 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 14:15:00 | 415.00 | 413.32 | 413.12 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 411.70 | 413.10 | 413.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 13:15:00 | 410.45 | 412.55 | 412.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 14:15:00 | 412.65 | 412.57 | 412.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 14:15:00 | 412.65 | 412.57 | 412.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 412.65 | 412.57 | 412.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 15:00:00 | 412.65 | 412.57 | 412.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 406.00 | 411.26 | 412.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 10:45:00 | 405.00 | 409.87 | 411.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:00:00 | 404.90 | 408.88 | 410.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 09:15:00 | 414.55 | 409.12 | 410.03 | SL hit (close>static) qty=1.00 sl=412.95 alert=retest2 |

### Cycle 20 — BUY (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 11:15:00 | 414.05 | 411.03 | 410.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 11:15:00 | 418.05 | 415.20 | 413.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 407.75 | 414.15 | 413.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 407.75 | 414.15 | 413.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 407.75 | 414.15 | 413.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 407.75 | 414.15 | 413.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 411.35 | 413.59 | 413.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 414.20 | 413.15 | 412.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-19 10:15:00 | 455.62 | 436.01 | 430.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 472.65 | 473.77 | 473.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 467.60 | 472.29 | 473.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 13:15:00 | 471.30 | 470.75 | 471.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 13:15:00 | 471.30 | 470.75 | 471.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 471.30 | 470.75 | 471.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:00:00 | 471.30 | 470.75 | 471.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 474.10 | 471.42 | 472.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 474.10 | 471.42 | 472.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 473.00 | 471.74 | 472.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 479.15 | 471.74 | 472.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 485.00 | 474.39 | 473.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 09:15:00 | 489.35 | 484.75 | 481.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 481.55 | 484.48 | 482.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 11:15:00 | 481.55 | 484.48 | 482.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 481.55 | 484.48 | 482.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 483.55 | 484.48 | 482.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 472.10 | 482.00 | 481.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 472.10 | 482.00 | 481.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 467.70 | 479.14 | 480.10 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 481.55 | 479.65 | 479.59 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 10:15:00 | 477.70 | 479.27 | 479.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 12:15:00 | 475.75 | 478.45 | 479.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 09:15:00 | 485.05 | 478.86 | 478.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 485.05 | 478.86 | 478.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 485.05 | 478.86 | 478.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:00:00 | 485.05 | 478.86 | 478.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 484.00 | 479.89 | 479.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 15:15:00 | 485.40 | 483.15 | 481.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 483.50 | 483.51 | 481.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 483.50 | 483.51 | 481.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 483.50 | 483.51 | 481.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:30:00 | 483.20 | 483.51 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 483.85 | 483.92 | 482.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 487.65 | 482.80 | 482.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 13:00:00 | 494.80 | 487.49 | 484.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 14:30:00 | 487.50 | 489.80 | 488.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 12:00:00 | 486.20 | 487.51 | 487.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 12:15:00 | 486.05 | 487.22 | 487.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 486.05 | 487.22 | 487.35 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 14:15:00 | 488.00 | 487.44 | 487.44 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 487.00 | 487.35 | 487.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 481.30 | 486.14 | 486.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 487.75 | 484.72 | 485.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 13:15:00 | 487.75 | 484.72 | 485.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 487.75 | 484.72 | 485.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:00:00 | 487.75 | 484.72 | 485.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 486.25 | 485.03 | 485.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 486.25 | 485.03 | 485.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 485.50 | 485.12 | 485.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:15:00 | 485.80 | 485.12 | 485.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 486.95 | 485.49 | 485.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 14:45:00 | 479.75 | 482.88 | 484.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 15:15:00 | 479.50 | 482.88 | 484.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 12:15:00 | 479.75 | 480.47 | 481.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 12:45:00 | 479.55 | 480.16 | 481.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 479.90 | 479.84 | 480.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 479.90 | 479.84 | 480.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 480.90 | 479.60 | 480.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:15:00 | 483.50 | 479.60 | 480.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 478.70 | 479.42 | 480.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:30:00 | 483.00 | 479.42 | 480.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 480.90 | 479.61 | 480.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:00:00 | 480.90 | 479.61 | 480.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 481.25 | 479.94 | 480.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:45:00 | 481.50 | 479.94 | 480.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 481.00 | 480.15 | 480.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 481.00 | 480.15 | 480.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 481.00 | 480.32 | 480.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 487.80 | 480.32 | 480.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 494.00 | 483.05 | 481.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 494.00 | 483.05 | 481.66 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 481.50 | 484.00 | 484.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 13:15:00 | 480.85 | 482.03 | 483.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 483.05 | 478.76 | 480.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 483.05 | 478.76 | 480.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 483.05 | 478.76 | 480.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:45:00 | 482.00 | 478.76 | 480.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 486.95 | 480.40 | 480.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 486.95 | 480.40 | 480.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 483.60 | 481.04 | 480.99 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 481.25 | 482.12 | 482.17 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 10:15:00 | 484.70 | 481.84 | 481.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 11:15:00 | 493.00 | 484.07 | 482.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 09:15:00 | 487.30 | 490.81 | 487.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 487.30 | 490.81 | 487.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 487.30 | 490.81 | 487.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:00:00 | 487.30 | 490.81 | 487.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 485.55 | 489.76 | 487.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:15:00 | 485.30 | 489.76 | 487.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 489.55 | 489.72 | 487.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:30:00 | 485.55 | 489.72 | 487.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 486.30 | 489.77 | 488.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 15:00:00 | 486.30 | 489.77 | 488.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 488.95 | 489.61 | 488.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 486.65 | 489.61 | 488.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 483.50 | 488.39 | 487.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:00:00 | 483.50 | 488.39 | 487.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 10:15:00 | 481.90 | 487.09 | 487.15 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 495.00 | 486.70 | 486.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 14:15:00 | 514.00 | 494.03 | 489.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 14:15:00 | 514.00 | 514.49 | 508.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 11:15:00 | 517.00 | 514.70 | 510.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 519.65 | 521.56 | 518.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 13:30:00 | 519.25 | 521.56 | 518.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 14:15:00 | 520.00 | 521.25 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 14:30:00 | 518.30 | 521.25 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 506.00 | 518.30 | 517.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 506.00 | 518.30 | 517.58 | SL hit (close<ema400) qty=1.00 sl=517.58 alert=retest1 |

### Cycle 37 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 507.55 | 516.15 | 516.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 481.50 | 500.74 | 508.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 495.70 | 492.99 | 499.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 495.70 | 492.99 | 499.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 495.70 | 492.99 | 499.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 501.90 | 492.99 | 499.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 498.50 | 492.20 | 495.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 14:30:00 | 492.50 | 495.04 | 495.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 492.15 | 494.56 | 495.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 492.85 | 487.25 | 487.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 492.85 | 487.25 | 487.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 11:15:00 | 496.00 | 490.53 | 488.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 12:15:00 | 490.25 | 490.47 | 488.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 12:45:00 | 490.75 | 490.47 | 488.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 488.40 | 490.19 | 489.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 488.40 | 490.19 | 489.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 489.50 | 490.05 | 489.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 490.00 | 490.05 | 489.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 487.95 | 489.63 | 489.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:00:00 | 487.95 | 489.63 | 489.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 485.95 | 488.90 | 488.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 485.60 | 488.90 | 488.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 487.60 | 488.64 | 488.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:30:00 | 485.05 | 488.64 | 488.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 12:15:00 | 485.20 | 487.95 | 488.32 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 492.35 | 489.00 | 488.69 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 487.95 | 488.55 | 488.58 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 14:15:00 | 491.25 | 489.09 | 488.82 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 486.30 | 488.56 | 488.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 14:15:00 | 484.15 | 487.06 | 487.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 09:15:00 | 489.00 | 486.97 | 487.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 489.00 | 486.97 | 487.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 489.00 | 486.97 | 487.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:45:00 | 492.45 | 486.97 | 487.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 488.60 | 487.30 | 487.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:30:00 | 490.00 | 487.30 | 487.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 492.00 | 488.24 | 488.16 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 15:15:00 | 487.10 | 488.15 | 488.19 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 10:15:00 | 489.95 | 488.52 | 488.35 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 483.60 | 487.54 | 487.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 480.00 | 486.03 | 487.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 484.35 | 483.59 | 485.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 484.35 | 483.59 | 485.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 484.35 | 483.59 | 485.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:30:00 | 482.00 | 483.52 | 485.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 13:15:00 | 482.10 | 483.43 | 484.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 09:15:00 | 482.35 | 482.99 | 484.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 10:30:00 | 481.15 | 482.08 | 483.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 473.40 | 473.29 | 476.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:45:00 | 476.65 | 473.29 | 476.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 473.95 | 472.67 | 475.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 11:00:00 | 472.90 | 472.71 | 475.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 11:30:00 | 472.85 | 472.65 | 474.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:00:00 | 472.40 | 472.65 | 474.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 14:30:00 | 472.10 | 472.64 | 474.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 476.45 | 473.21 | 474.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:30:00 | 476.80 | 473.21 | 474.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 485.50 | 475.67 | 475.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 485.50 | 475.67 | 475.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 13:15:00 | 491.60 | 486.05 | 484.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 11:15:00 | 486.80 | 488.35 | 486.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 11:15:00 | 486.80 | 488.35 | 486.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 486.80 | 488.35 | 486.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:45:00 | 486.80 | 488.35 | 486.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 487.75 | 488.56 | 487.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:15:00 | 488.80 | 488.56 | 487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 486.55 | 488.16 | 487.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:00:00 | 486.55 | 488.16 | 487.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 484.05 | 487.33 | 486.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:45:00 | 487.10 | 487.33 | 486.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 483.95 | 486.66 | 486.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:45:00 | 483.00 | 486.66 | 486.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 485.00 | 486.33 | 486.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 10:15:00 | 481.65 | 484.86 | 485.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 485.75 | 485.04 | 485.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 485.75 | 485.04 | 485.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 485.75 | 485.04 | 485.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:00:00 | 485.75 | 485.04 | 485.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 12:15:00 | 490.60 | 486.15 | 486.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 11:15:00 | 493.25 | 490.14 | 488.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 12:15:00 | 489.00 | 489.91 | 488.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-20 13:00:00 | 489.00 | 489.91 | 488.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 485.50 | 488.84 | 488.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 491.05 | 488.84 | 488.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 481.45 | 487.36 | 487.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 481.45 | 487.36 | 487.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 480.90 | 485.55 | 486.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 467.00 | 461.38 | 467.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 467.00 | 461.38 | 467.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 467.00 | 461.38 | 467.42 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 474.40 | 469.15 | 468.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 09:15:00 | 484.95 | 475.13 | 472.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 11:15:00 | 474.80 | 476.35 | 473.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:00:00 | 474.80 | 476.35 | 473.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 471.75 | 475.43 | 473.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 471.75 | 475.43 | 473.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 470.45 | 474.43 | 473.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:00:00 | 470.45 | 474.43 | 473.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 473.65 | 473.76 | 473.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:45:00 | 473.20 | 473.76 | 473.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 473.10 | 473.62 | 473.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:45:00 | 473.60 | 473.62 | 473.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 472.35 | 473.37 | 473.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 472.35 | 473.37 | 473.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 473.70 | 473.44 | 473.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:30:00 | 472.00 | 473.44 | 473.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 475.25 | 473.80 | 473.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 478.25 | 473.80 | 473.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-09 09:15:00 | 526.08 | 519.27 | 510.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 10:15:00 | 540.55 | 546.92 | 547.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 537.35 | 545.01 | 546.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 541.50 | 539.45 | 542.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 541.50 | 539.45 | 542.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 541.50 | 539.45 | 542.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:45:00 | 540.80 | 539.45 | 542.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 541.00 | 539.76 | 542.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:30:00 | 538.90 | 540.27 | 541.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:00:00 | 538.95 | 540.27 | 541.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 547.75 | 541.57 | 542.11 | SL hit (close>static) qty=1.00 sl=543.00 alert=retest2 |

### Cycle 54 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 548.15 | 542.88 | 542.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 551.30 | 547.52 | 545.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 10:15:00 | 547.05 | 547.42 | 545.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 10:15:00 | 547.05 | 547.42 | 545.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 547.05 | 547.42 | 545.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 547.05 | 547.42 | 545.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 544.45 | 546.83 | 545.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:30:00 | 544.00 | 546.83 | 545.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 541.55 | 545.77 | 545.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 12:45:00 | 541.45 | 545.77 | 545.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 542.50 | 544.39 | 544.58 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 545.30 | 544.78 | 544.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 551.20 | 546.31 | 545.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 555.20 | 555.25 | 552.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 14:15:00 | 555.20 | 555.25 | 552.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 555.20 | 555.25 | 552.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 15:00:00 | 555.20 | 555.25 | 552.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 553.00 | 554.80 | 552.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 09:15:00 | 553.55 | 554.80 | 552.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 548.20 | 553.48 | 552.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 10:00:00 | 548.20 | 553.48 | 552.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 547.40 | 552.27 | 551.65 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2023-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 11:15:00 | 546.05 | 551.02 | 551.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 14:15:00 | 537.80 | 547.18 | 549.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 14:15:00 | 543.00 | 542.25 | 545.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-05 15:00:00 | 543.00 | 542.25 | 545.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 545.00 | 542.28 | 544.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:00:00 | 545.00 | 542.28 | 544.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 547.60 | 543.34 | 544.45 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 546.45 | 545.37 | 545.22 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 09:15:00 | 539.00 | 544.09 | 544.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 12:15:00 | 537.00 | 541.54 | 543.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 15:15:00 | 530.00 | 528.68 | 533.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 09:15:00 | 523.20 | 528.68 | 533.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 15:15:00 | 526.00 | 525.61 | 529.33 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 525.00 | 525.55 | 528.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-12 10:15:00 | 530.30 | 526.50 | 528.81 | SL hit (close>ema400) qty=1.00 sl=528.81 alert=retest1 |

### Cycle 60 — BUY (started 2023-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 13:15:00 | 535.00 | 530.70 | 530.38 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 526.05 | 529.71 | 530.07 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 12:15:00 | 532.30 | 530.27 | 530.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 535.00 | 531.22 | 530.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 544.00 | 544.34 | 540.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 12:15:00 | 539.95 | 543.04 | 541.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 539.95 | 543.04 | 541.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:30:00 | 540.00 | 543.04 | 541.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 539.45 | 542.33 | 540.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:45:00 | 540.50 | 542.33 | 540.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 537.50 | 541.36 | 540.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 15:00:00 | 537.50 | 541.36 | 540.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 538.50 | 540.79 | 540.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 538.80 | 540.79 | 540.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 12:00:00 | 540.10 | 542.53 | 542.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 535.00 | 540.62 | 541.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 535.00 | 540.62 | 541.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 531.90 | 537.43 | 539.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 13:15:00 | 535.00 | 534.80 | 537.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 13:45:00 | 535.60 | 534.80 | 537.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 539.60 | 535.76 | 537.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:45:00 | 541.00 | 535.76 | 537.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 536.75 | 535.96 | 537.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 545.70 | 535.96 | 537.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 544.05 | 537.58 | 538.11 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 547.00 | 539.46 | 538.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 554.40 | 545.19 | 542.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 15:15:00 | 558.00 | 558.35 | 553.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 555.45 | 557.77 | 553.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 555.45 | 557.77 | 553.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:00:00 | 555.45 | 557.77 | 553.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 554.65 | 556.72 | 553.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:45:00 | 554.15 | 556.72 | 553.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 552.35 | 556.17 | 554.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 552.35 | 556.17 | 554.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 553.60 | 555.66 | 554.62 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 549.00 | 554.11 | 554.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 14:15:00 | 545.95 | 552.48 | 553.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-01 12:15:00 | 550.75 | 549.69 | 551.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 12:15:00 | 550.75 | 549.69 | 551.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 550.75 | 549.69 | 551.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:45:00 | 551.05 | 549.69 | 551.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 553.75 | 550.50 | 551.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 14:00:00 | 553.75 | 550.50 | 551.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 547.15 | 549.83 | 551.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 10:15:00 | 545.75 | 548.90 | 550.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 554.45 | 550.22 | 550.71 | SL hit (close>static) qty=1.00 sl=553.80 alert=retest2 |

### Cycle 66 — BUY (started 2024-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 13:15:00 | 555.80 | 551.33 | 551.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 567.90 | 554.98 | 553.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 576.50 | 577.79 | 573.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 09:45:00 | 575.60 | 577.79 | 573.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 573.10 | 576.85 | 573.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:30:00 | 572.80 | 576.85 | 573.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 573.00 | 576.08 | 573.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:00:00 | 573.00 | 576.08 | 573.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 572.80 | 575.42 | 573.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 13:45:00 | 576.50 | 575.54 | 573.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 11:45:00 | 573.60 | 573.84 | 573.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:45:00 | 574.75 | 573.64 | 573.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 569.45 | 572.70 | 572.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 09:15:00 | 569.45 | 572.70 | 572.93 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 581.60 | 573.59 | 573.01 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 569.85 | 578.06 | 578.57 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 579.90 | 577.73 | 577.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-23 09:15:00 | 583.00 | 578.77 | 578.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 578.50 | 578.72 | 578.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 10:15:00 | 578.50 | 578.72 | 578.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 578.50 | 578.72 | 578.06 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 574.90 | 577.26 | 577.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 573.00 | 576.41 | 577.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 573.85 | 572.69 | 574.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:00:00 | 573.85 | 572.69 | 574.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 579.90 | 574.13 | 574.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:45:00 | 578.90 | 574.13 | 574.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 576.30 | 574.56 | 575.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:30:00 | 579.70 | 574.56 | 575.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 576.00 | 574.85 | 575.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 581.50 | 574.85 | 575.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 578.20 | 575.52 | 575.45 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 572.35 | 574.89 | 575.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 570.60 | 574.03 | 574.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 14:15:00 | 573.85 | 573.79 | 574.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 14:15:00 | 573.85 | 573.79 | 574.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 573.85 | 573.79 | 574.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:15:00 | 573.80 | 573.79 | 574.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 573.80 | 573.79 | 574.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 578.80 | 573.79 | 574.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 576.30 | 574.29 | 574.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:30:00 | 579.20 | 574.29 | 574.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 578.75 | 575.19 | 574.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 583.20 | 577.61 | 576.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 12:15:00 | 581.40 | 584.52 | 581.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 12:15:00 | 581.40 | 584.52 | 581.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 581.40 | 584.52 | 581.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:00:00 | 581.40 | 584.52 | 581.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 583.00 | 584.22 | 581.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:45:00 | 580.90 | 584.22 | 581.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 580.60 | 583.49 | 581.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:45:00 | 585.25 | 583.49 | 581.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 581.25 | 583.05 | 581.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:15:00 | 595.20 | 583.05 | 581.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 11:15:00 | 579.50 | 586.38 | 585.92 | SL hit (close<static) qty=1.00 sl=580.15 alert=retest2 |

### Cycle 75 — SELL (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 12:15:00 | 581.90 | 585.49 | 585.55 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 592.00 | 586.33 | 585.80 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 542.50 | 578.82 | 582.86 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 12:15:00 | 568.35 | 564.21 | 563.72 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 551.70 | 561.87 | 562.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 547.05 | 557.94 | 560.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 14:15:00 | 548.00 | 547.96 | 552.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 14:45:00 | 549.50 | 547.96 | 552.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 533.20 | 529.85 | 535.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:30:00 | 534.95 | 529.85 | 535.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 544.90 | 532.86 | 536.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 543.50 | 532.86 | 536.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 541.35 | 534.56 | 536.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 15:00:00 | 539.65 | 537.86 | 537.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 15:15:00 | 539.00 | 538.08 | 538.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 15:15:00 | 539.00 | 538.08 | 538.02 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 13:15:00 | 537.25 | 537.89 | 537.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 533.40 | 536.70 | 537.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 537.60 | 536.22 | 537.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 11:15:00 | 537.60 | 536.22 | 537.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 537.60 | 536.22 | 537.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:00:00 | 537.60 | 536.22 | 537.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 12:15:00 | 544.00 | 537.77 | 537.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 15:15:00 | 546.00 | 540.98 | 539.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 536.20 | 540.02 | 539.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 536.20 | 540.02 | 539.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 536.20 | 540.02 | 539.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:45:00 | 535.65 | 540.02 | 539.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 535.40 | 539.10 | 538.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 536.30 | 539.10 | 538.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 537.00 | 538.73 | 538.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 13:00:00 | 537.00 | 538.73 | 538.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 13:15:00 | 536.55 | 538.29 | 538.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 14:15:00 | 535.00 | 537.63 | 538.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 534.10 | 533.75 | 535.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 11:45:00 | 533.60 | 533.75 | 535.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 526.95 | 528.67 | 530.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 522.55 | 527.27 | 528.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 14:45:00 | 522.50 | 521.12 | 524.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 520.55 | 521.28 | 522.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-02 12:00:00 | 521.50 | 520.48 | 520.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 518.75 | 520.54 | 520.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 10:30:00 | 517.55 | 519.78 | 520.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:30:00 | 513.90 | 508.86 | 512.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 15:15:00 | 519.00 | 513.51 | 513.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 15:15:00 | 519.00 | 513.51 | 513.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 10:15:00 | 519.80 | 514.99 | 514.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 526.00 | 528.02 | 523.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 15:15:00 | 523.00 | 528.02 | 523.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 523.00 | 527.02 | 523.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:15:00 | 513.75 | 527.02 | 523.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 515.30 | 524.67 | 522.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:30:00 | 514.25 | 524.67 | 522.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 511.80 | 522.10 | 521.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:00:00 | 511.80 | 522.10 | 521.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 510.00 | 519.68 | 520.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 496.80 | 510.99 | 515.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 15:15:00 | 504.95 | 504.38 | 509.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-14 09:15:00 | 498.05 | 504.38 | 509.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 490.30 | 487.14 | 491.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:00:00 | 490.30 | 487.14 | 491.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 487.15 | 487.14 | 491.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:30:00 | 489.25 | 487.14 | 491.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 488.25 | 487.14 | 490.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:30:00 | 483.30 | 486.35 | 489.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 09:45:00 | 482.25 | 484.44 | 487.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 11:15:00 | 490.65 | 485.13 | 487.07 | SL hit (close>ema400) qty=1.00 sl=487.07 alert=retest1 |

### Cycle 86 — BUY (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 15:15:00 | 490.75 | 488.60 | 488.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 492.90 | 489.59 | 488.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 11:15:00 | 492.90 | 495.13 | 492.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 11:15:00 | 492.90 | 495.13 | 492.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 492.90 | 495.13 | 492.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 12:00:00 | 492.90 | 495.13 | 492.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 493.55 | 494.81 | 493.02 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 09:15:00 | 489.70 | 492.10 | 492.15 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 495.75 | 489.61 | 489.46 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 09:15:00 | 485.15 | 489.30 | 489.40 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 14:15:00 | 490.30 | 489.46 | 489.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 15:15:00 | 491.90 | 489.95 | 489.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 12:15:00 | 533.85 | 537.57 | 530.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 13:00:00 | 533.85 | 537.57 | 530.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 540.10 | 538.13 | 532.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 09:30:00 | 541.30 | 538.13 | 532.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 541.00 | 539.77 | 536.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 544.00 | 539.77 | 536.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 527.40 | 543.22 | 544.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 527.40 | 543.22 | 544.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 520.05 | 524.62 | 528.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 10:15:00 | 523.80 | 523.79 | 527.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 11:00:00 | 523.80 | 523.79 | 527.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 528.70 | 524.89 | 527.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:45:00 | 528.40 | 524.89 | 527.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 525.60 | 525.03 | 527.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:15:00 | 524.00 | 525.03 | 527.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 11:00:00 | 523.00 | 521.06 | 521.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 530.00 | 522.85 | 522.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 530.00 | 522.85 | 522.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 536.50 | 527.07 | 524.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 10:15:00 | 529.90 | 530.38 | 528.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 10:15:00 | 529.90 | 530.38 | 528.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 529.90 | 530.38 | 528.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:30:00 | 529.05 | 530.38 | 528.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 526.65 | 529.53 | 528.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 529.70 | 528.88 | 528.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 532.75 | 528.28 | 528.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:30:00 | 531.00 | 530.24 | 530.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:45:00 | 531.80 | 530.39 | 530.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 15:15:00 | 530.00 | 530.31 | 530.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 15:15:00 | 530.00 | 530.31 | 530.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 10:15:00 | 525.80 | 529.03 | 529.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 15:15:00 | 520.00 | 519.90 | 523.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:15:00 | 521.40 | 519.90 | 523.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 524.15 | 520.75 | 523.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:00:00 | 514.10 | 519.91 | 521.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 11:15:00 | 530.70 | 520.00 | 520.40 | SL hit (close>static) qty=1.00 sl=526.30 alert=retest2 |

### Cycle 94 — BUY (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 12:15:00 | 525.10 | 521.02 | 520.83 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 12:15:00 | 520.05 | 522.49 | 522.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 13:15:00 | 517.60 | 521.51 | 522.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 521.80 | 519.67 | 520.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 517.95 | 519.32 | 520.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 534.00 | 519.32 | 520.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 547.35 | 524.93 | 522.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 551.25 | 530.19 | 525.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 545.15 | 545.88 | 541.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 11:30:00 | 547.00 | 545.88 | 541.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 542.00 | 544.93 | 541.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 541.05 | 544.93 | 541.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 539.40 | 543.83 | 541.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 539.40 | 543.83 | 541.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 539.20 | 542.90 | 541.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 539.20 | 542.90 | 541.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 534.50 | 539.59 | 539.93 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 566.40 | 542.37 | 539.96 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 549.45 | 551.45 | 551.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 544.00 | 549.09 | 550.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 545.05 | 544.59 | 547.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 545.05 | 544.59 | 547.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 561.00 | 547.94 | 548.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:15:00 | 571.50 | 547.94 | 548.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 585.35 | 555.42 | 551.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 597.40 | 572.26 | 561.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 10:15:00 | 579.80 | 580.63 | 569.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 10:45:00 | 580.40 | 580.63 | 569.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 572.35 | 576.80 | 572.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 575.95 | 576.80 | 572.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 551.05 | 568.96 | 569.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 551.05 | 568.96 | 569.26 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 571.65 | 569.50 | 569.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 577.50 | 571.10 | 570.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 682.30 | 683.36 | 663.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:45:00 | 684.55 | 683.36 | 663.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 722.05 | 728.67 | 723.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 720.25 | 728.67 | 723.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 721.25 | 727.18 | 723.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 721.25 | 727.18 | 723.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 727.95 | 727.34 | 723.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 732.20 | 724.40 | 723.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 13:15:00 | 718.70 | 724.20 | 723.82 | SL hit (close<static) qty=1.00 sl=721.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 720.80 | 723.52 | 723.54 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 725.80 | 723.52 | 723.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 727.35 | 724.29 | 723.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 717.85 | 725.48 | 724.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 722.00 | 724.79 | 724.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:15:00 | 722.80 | 724.79 | 724.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 721.05 | 724.04 | 724.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 721.05 | 724.04 | 724.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 719.60 | 722.71 | 723.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 725.65 | 723.29 | 723.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 723.00 | 723.24 | 723.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 721.45 | 723.24 | 723.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 720.35 | 723.33 | 723.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 721.55 | 720.63 | 721.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 728.95 | 723.33 | 722.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 728.95 | 723.33 | 722.74 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 719.25 | 723.13 | 723.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 716.60 | 719.20 | 720.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 724.40 | 719.35 | 720.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 725.00 | 720.48 | 721.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 725.00 | 720.48 | 721.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 722.10 | 721.45 | 721.39 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 720.10 | 721.18 | 721.27 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 724.10 | 721.41 | 721.32 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 714.25 | 723.06 | 723.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 712.00 | 718.50 | 721.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 714.70 | 717.10 | 719.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 714.85 | 717.10 | 719.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 702.80 | 698.94 | 698.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 702.80 | 698.94 | 698.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 704.40 | 700.04 | 698.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 708.00 | 708.11 | 704.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:15:00 | 708.65 | 708.11 | 704.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 704.70 | 707.43 | 704.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 704.55 | 707.43 | 704.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 711.90 | 708.33 | 705.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 718.55 | 708.38 | 707.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 703.25 | 707.35 | 707.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 703.25 | 707.35 | 707.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 699.85 | 705.85 | 706.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 703.85 | 705.06 | 706.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 704.10 | 704.87 | 705.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 700.00 | 704.87 | 705.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:45:00 | 698.35 | 702.13 | 704.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 706.80 | 702.13 | 702.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 706.80 | 702.13 | 702.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 713.00 | 708.04 | 705.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 706.85 | 707.99 | 705.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 709.65 | 708.32 | 706.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 706.15 | 708.32 | 706.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 714.50 | 718.23 | 713.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 714.50 | 718.23 | 713.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 711.60 | 716.91 | 713.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 711.60 | 716.91 | 713.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 710.75 | 715.68 | 713.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:15:00 | 712.45 | 715.68 | 713.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:00:00 | 712.65 | 714.48 | 713.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 703.40 | 711.25 | 712.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 09:15:00 | 703.40 | 711.25 | 712.00 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 716.10 | 712.57 | 712.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 719.95 | 715.29 | 713.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 708.05 | 712.35 | 712.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 706.55 | 709.51 | 711.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 712.60 | 710.30 | 711.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 713.30 | 710.90 | 711.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:15:00 | 726.80 | 710.90 | 711.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 12:15:00 | 723.00 | 713.32 | 712.52 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 709.60 | 714.29 | 714.56 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 717.00 | 714.83 | 714.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 11:15:00 | 772.00 | 726.27 | 719.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 704.35 | 720.95 | 722.67 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 734.00 | 722.54 | 721.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 742.30 | 726.49 | 723.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 849.00 | 854.93 | 838.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:30:00 | 847.55 | 854.93 | 838.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 847.10 | 852.56 | 840.35 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 851.50 | 856.14 | 856.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 848.35 | 853.12 | 854.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 847.10 | 853.51 | 854.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 848.05 | 853.51 | 854.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:45:00 | 848.00 | 852.41 | 853.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 861.50 | 854.91 | 854.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 861.50 | 854.91 | 854.50 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 848.30 | 855.50 | 855.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 840.75 | 851.48 | 853.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 13:15:00 | 833.90 | 833.28 | 840.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 14:00:00 | 833.90 | 833.28 | 840.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 820.90 | 831.23 | 837.49 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 853.75 | 838.37 | 836.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 868.75 | 849.47 | 843.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:45:00 | 862.30 | 869.57 | 860.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 865.85 | 874.83 | 869.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 865.85 | 874.83 | 869.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 868.55 | 873.57 | 869.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:15:00 | 863.60 | 873.57 | 869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 871.40 | 873.14 | 869.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:30:00 | 870.45 | 873.14 | 869.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 865.35 | 871.58 | 869.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:45:00 | 865.05 | 871.58 | 869.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 862.10 | 869.68 | 868.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:45:00 | 862.70 | 869.68 | 868.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 860.25 | 866.66 | 867.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 837.00 | 860.72 | 864.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:00:00 | 857.00 | 851.40 | 853.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 862.00 | 855.43 | 855.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 862.00 | 855.43 | 855.28 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 849.75 | 855.16 | 855.24 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 864.70 | 856.54 | 855.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 867.40 | 858.71 | 856.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 871.20 | 871.60 | 865.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:45:00 | 869.50 | 871.60 | 865.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 860.00 | 872.18 | 868.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 860.00 | 872.18 | 868.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 855.30 | 868.81 | 867.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 855.30 | 868.81 | 867.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 853.50 | 865.75 | 866.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 13:15:00 | 849.05 | 860.38 | 863.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 856.25 | 855.43 | 859.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 856.25 | 855.43 | 859.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 860.90 | 856.52 | 859.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 860.90 | 856.52 | 859.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 859.20 | 857.06 | 859.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 861.50 | 857.06 | 859.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 866.05 | 858.86 | 860.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 866.05 | 858.86 | 860.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 862.90 | 859.67 | 860.59 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 868.00 | 861.33 | 861.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 907.95 | 870.66 | 865.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 11:15:00 | 925.10 | 925.22 | 911.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:30:00 | 925.35 | 925.22 | 911.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 964.90 | 935.35 | 919.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 964.90 | 935.35 | 919.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 954.00 | 968.94 | 954.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 946.75 | 968.94 | 954.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 956.35 | 966.42 | 955.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:15:00 | 962.00 | 966.42 | 955.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 961.40 | 965.42 | 955.58 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 942.85 | 951.28 | 951.85 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 969.90 | 955.50 | 953.54 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 950.50 | 963.23 | 964.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 946.00 | 959.78 | 962.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:00:00 | 918.90 | 935.16 | 943.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:45:00 | 919.15 | 932.09 | 940.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 14:45:00 | 921.10 | 929.67 | 939.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 872.95 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 873.19 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 875.04 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 862.35 | 862.26 | 878.71 | SL hit (close>ema200) qty=0.50 sl=862.26 alert=retest2 |

### Cycle 136 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 859.80 | 855.83 | 855.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 864.05 | 858.76 | 857.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 12:15:00 | 899.40 | 900.82 | 889.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 12:45:00 | 899.85 | 900.82 | 889.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 916.65 | 905.15 | 895.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:30:00 | 918.55 | 910.66 | 900.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 885.00 | 897.95 | 899.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 885.00 | 897.95 | 899.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 876.60 | 892.42 | 896.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 858.90 | 855.10 | 870.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 858.90 | 855.10 | 870.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 850.00 | 854.76 | 863.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 840.00 | 851.98 | 859.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 870.25 | 851.67 | 856.79 | SL hit (close>static) qty=1.00 sl=867.50 alert=retest2 |

### Cycle 138 — BUY (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 14:15:00 | 867.40 | 860.24 | 859.63 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 840.05 | 856.95 | 859.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 838.00 | 850.93 | 855.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 854.85 | 845.45 | 848.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 850.70 | 846.50 | 849.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 853.90 | 846.50 | 849.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 850.00 | 847.20 | 849.21 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 855.00 | 850.11 | 849.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 857.80 | 851.64 | 850.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 875.00 | 880.53 | 874.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 874.25 | 879.27 | 874.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 875.45 | 879.27 | 874.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 872.60 | 877.94 | 874.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 872.60 | 877.94 | 874.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 874.90 | 877.33 | 874.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 872.20 | 877.33 | 874.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 874.10 | 876.68 | 874.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 873.25 | 876.68 | 874.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 877.35 | 876.82 | 874.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 874.30 | 876.82 | 874.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 880.75 | 877.60 | 875.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 877.05 | 877.60 | 875.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 879.95 | 878.07 | 875.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 876.00 | 878.07 | 875.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 873.75 | 877.04 | 875.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 871.35 | 877.04 | 875.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 870.00 | 875.63 | 875.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 867.95 | 875.63 | 875.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 869.45 | 874.39 | 874.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 862.40 | 870.44 | 872.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 866.90 | 866.48 | 869.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 866.90 | 866.48 | 869.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 866.30 | 862.27 | 865.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 860.80 | 861.34 | 865.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 817.76 | 828.90 | 841.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-18 09:15:00 | 774.72 | 799.40 | 817.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 142 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 768.45 | 752.52 | 750.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 771.40 | 759.07 | 754.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 802.45 | 802.89 | 790.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 802.45 | 802.89 | 790.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 795.50 | 801.40 | 793.86 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 15:15:00 | 792.65 | 794.63 | 794.90 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 808.15 | 797.33 | 796.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 829.05 | 806.96 | 801.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 814.60 | 816.31 | 808.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 14:00:00 | 814.60 | 816.31 | 808.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 819.30 | 815.92 | 810.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:30:00 | 832.50 | 817.13 | 812.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 825.05 | 819.99 | 816.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 824.50 | 819.99 | 816.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 12:15:00 | 823.50 | 838.75 | 840.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 823.50 | 838.75 | 840.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 816.45 | 834.29 | 838.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 810.60 | 802.36 | 810.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 811.00 | 804.09 | 810.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 808.30 | 804.09 | 810.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 800.65 | 803.40 | 809.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 797.50 | 801.31 | 805.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 789.35 | 798.94 | 804.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:15:00 | 757.62 | 778.43 | 789.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 777.05 | 767.85 | 778.16 | SL hit (close>ema200) qty=0.50 sl=767.85 alert=retest2 |

### Cycle 146 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 785.50 | 776.59 | 776.14 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 768.85 | 776.13 | 776.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 12:15:00 | 767.70 | 772.51 | 774.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 772.30 | 767.22 | 769.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 774.95 | 768.77 | 770.21 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 776.95 | 771.71 | 771.32 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 759.30 | 770.49 | 771.10 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 779.20 | 772.02 | 771.41 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 10:15:00 | 765.05 | 770.87 | 771.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 11:15:00 | 753.10 | 767.31 | 769.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 738.50 | 733.67 | 740.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 735.30 | 733.99 | 740.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:00:00 | 732.05 | 733.61 | 739.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 695.45 | 709.68 | 716.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 690.00 | 689.99 | 696.86 | SL hit (close>ema200) qty=0.50 sl=689.99 alert=retest2 |

### Cycle 152 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 701.50 | 698.97 | 698.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 711.00 | 701.38 | 700.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 710.05 | 710.92 | 706.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 710.05 | 710.92 | 706.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 699.65 | 708.34 | 706.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 699.65 | 708.34 | 706.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 702.55 | 707.18 | 706.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 699.55 | 707.18 | 706.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 699.00 | 705.02 | 705.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 10:15:00 | 696.30 | 703.28 | 704.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 699.30 | 696.62 | 699.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 699.35 | 697.17 | 699.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:45:00 | 698.40 | 697.17 | 699.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 696.00 | 696.93 | 699.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 684.65 | 697.33 | 699.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:00:00 | 692.70 | 691.89 | 693.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 692.10 | 691.95 | 693.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 691.80 | 692.26 | 693.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 686.10 | 691.03 | 692.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 685.65 | 689.44 | 691.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 667.30 | 688.63 | 690.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 658.07 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 657.50 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 657.21 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 650.42 | 665.90 | 676.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 651.37 | 665.90 | 676.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 633.93 | 653.55 | 667.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 650.05 | 648.10 | 658.07 | SL hit (close>ema200) qty=0.50 sl=648.10 alert=retest2 |

### Cycle 154 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 685.95 | 663.19 | 660.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 690.35 | 681.31 | 673.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 725.10 | 728.51 | 716.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:15:00 | 720.95 | 728.51 | 716.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 719.95 | 723.34 | 718.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 15:15:00 | 722.00 | 723.34 | 718.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:00:00 | 723.20 | 724.19 | 720.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 722.25 | 725.10 | 722.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 657.00 | 716.72 | 721.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 657.00 | 716.72 | 721.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 637.50 | 647.61 | 668.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 603.60 | 603.20 | 617.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:15:00 | 600.50 | 603.20 | 617.78 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:45:00 | 600.95 | 602.85 | 616.30 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 11:30:00 | 600.70 | 602.43 | 614.88 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 12:00:00 | 600.75 | 602.43 | 614.88 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 593.00 | 598.13 | 607.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 616.35 | 598.13 | 607.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.48 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.90 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.66 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.71 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 583.00 | 582.94 | 593.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 583.00 | 582.94 | 593.20 | SL hit (close>ema200) qty=0.50 sl=582.94 alert=retest1 |

### Cycle 156 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 602.85 | 583.41 | 582.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 606.35 | 588.00 | 584.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 633.95 | 638.76 | 623.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 633.95 | 638.76 | 623.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 645.40 | 652.85 | 646.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 649.65 | 652.85 | 646.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 632.00 | 643.05 | 644.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 632.00 | 643.05 | 644.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 621.50 | 638.74 | 642.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 627.55 | 627.49 | 634.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 627.55 | 627.49 | 634.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 625.05 | 627.00 | 634.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 616.70 | 627.00 | 634.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:30:00 | 618.05 | 621.78 | 629.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 14:15:00 | 619.50 | 621.60 | 628.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 615.85 | 624.10 | 628.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 603.80 | 620.04 | 626.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 643.90 | 625.37 | 626.54 | SL hit (close>static) qty=1.00 sl=638.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 655.20 | 631.33 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 657.00 | 640.25 | 633.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 11:15:00 | 663.35 | 663.79 | 653.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 12:00:00 | 663.35 | 663.79 | 653.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 669.35 | 673.64 | 669.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 669.35 | 673.64 | 669.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 665.65 | 672.04 | 669.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 663.90 | 672.04 | 669.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 663.75 | 670.39 | 668.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 664.10 | 670.39 | 668.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 662.35 | 668.78 | 668.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 661.55 | 668.78 | 668.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 658.00 | 666.62 | 667.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 655.80 | 663.03 | 665.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 645.00 | 644.44 | 651.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 643.75 | 644.44 | 651.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 644.60 | 644.47 | 650.80 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 657.90 | 652.55 | 652.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 11:15:00 | 663.80 | 655.54 | 653.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 11:15:00 | 677.05 | 679.36 | 672.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:00:00 | 677.05 | 679.36 | 672.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 675.35 | 677.99 | 673.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 672.35 | 677.99 | 673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 694.65 | 697.67 | 694.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 692.75 | 697.67 | 694.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 697.00 | 697.53 | 694.45 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 12:15:00 | 689.60 | 693.44 | 693.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 15:15:00 | 686.50 | 690.88 | 692.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 666.45 | 664.45 | 671.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:45:00 | 667.95 | 664.45 | 671.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 667.25 | 662.07 | 666.86 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 691.95 | 669.79 | 668.86 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 14:15:00 | 670.10 | 675.33 | 675.68 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 680.10 | 675.73 | 675.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 681.90 | 678.39 | 676.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:00:00 | 689.60 | 680.68 | 678.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 669.00 | 677.25 | 677.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 669.00 | 677.25 | 677.30 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 14:15:00 | 678.00 | 677.40 | 677.36 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 673.00 | 676.52 | 676.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 638.00 | 668.82 | 673.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 675.20 | 658.60 | 665.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 672.00 | 661.28 | 665.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 685.50 | 661.28 | 665.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 680.75 | 668.49 | 668.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 695.00 | 680.15 | 676.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 711.65 | 712.35 | 705.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 711.65 | 712.35 | 705.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 708.70 | 711.41 | 707.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:45:00 | 720.15 | 712.62 | 708.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 716.45 | 713.49 | 709.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 13:15:00 | 731.25 | 734.67 | 734.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 731.25 | 734.67 | 734.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 728.55 | 733.45 | 734.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 708.05 | 704.79 | 711.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 690.90 | 696.18 | 700.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 690.90 | 696.18 | 700.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 688.20 | 694.58 | 699.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 688.70 | 686.81 | 690.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 706.10 | 691.34 | 691.86 | SL hit (close>static) qty=1.00 sl=704.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 701.40 | 693.35 | 692.72 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 684.15 | 691.35 | 692.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 680.90 | 689.26 | 691.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 701.00 | 688.38 | 686.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 704.15 | 691.53 | 688.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 692.15 | 699.33 | 695.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 693.05 | 698.07 | 694.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 690.35 | 698.07 | 694.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 690.10 | 696.48 | 694.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 690.10 | 696.48 | 694.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 704.85 | 709.22 | 705.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:45:00 | 710.60 | 707.89 | 705.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 711.50 | 716.10 | 716.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 711.50 | 716.10 | 716.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 709.60 | 713.90 | 715.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 721.65 | 713.73 | 714.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 724.55 | 715.90 | 715.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 734.00 | 729.94 | 726.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 757.55 | 759.89 | 750.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 757.55 | 759.89 | 750.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 762.45 | 764.99 | 758.29 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 753.85 | 759.30 | 759.75 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 763.50 | 760.12 | 759.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 765.00 | 761.75 | 760.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 753.90 | 761.50 | 760.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 756.85 | 760.57 | 760.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 760.35 | 760.57 | 760.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 760.00 | 760.66 | 760.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 760.00 | 760.66 | 760.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 760.70 | 760.67 | 760.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 760.70 | 760.67 | 760.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 759.65 | 760.47 | 760.54 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 763.75 | 761.12 | 760.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 765.70 | 762.04 | 761.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 755.20 | 761.97 | 761.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 758.00 | 761.17 | 761.10 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 758.15 | 760.57 | 760.83 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 763.00 | 760.62 | 760.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 768.25 | 762.53 | 761.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 757.10 | 761.95 | 761.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 753.05 | 760.17 | 760.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 746.65 | 757.46 | 759.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 737.65 | 740.74 | 744.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 700.77 | 712.86 | 722.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 711.75 | 711.35 | 719.08 | SL hit (close>ema200) qty=0.50 sl=711.35 alert=retest2 |

### Cycle 182 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 715.80 | 711.50 | 711.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 732.30 | 718.85 | 715.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 747.30 | 748.01 | 738.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 739.60 | 744.67 | 738.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 739.60 | 744.67 | 738.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 742.00 | 740.81 | 738.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 741.90 | 740.28 | 739.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 742.60 | 740.47 | 739.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 741.25 | 740.47 | 739.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 738.40 | 740.43 | 739.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 734.55 | 740.43 | 739.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 744.20 | 741.18 | 739.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 738.55 | 740.33 | 740.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 738.55 | 740.33 | 740.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 736.05 | 739.16 | 739.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 735.25 | 734.02 | 736.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 727.95 | 732.96 | 735.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 731.10 | 732.96 | 735.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 734.70 | 731.64 | 733.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 735.55 | 731.64 | 733.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 729.90 | 731.30 | 733.02 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 737.40 | 733.86 | 733.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 741.00 | 735.87 | 734.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 735.20 | 736.92 | 735.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 735.40 | 736.61 | 735.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 736.05 | 736.61 | 735.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 732.00 | 735.37 | 735.15 | SL hit (close<static) qty=1.00 sl=733.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 731.00 | 734.50 | 734.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 728.95 | 732.33 | 733.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 741.10 | 728.68 | 727.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 752.70 | 747.94 | 742.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 753.40 | 754.96 | 749.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 750.10 | 754.96 | 749.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 749.30 | 753.82 | 749.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 747.90 | 753.82 | 749.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 750.00 | 753.06 | 749.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 754.35 | 753.06 | 749.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 753.00 | 752.05 | 749.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 755.85 | 771.63 | 773.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 755.85 | 771.63 | 773.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 751.25 | 767.55 | 771.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 764.90 | 757.23 | 763.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 757.45 | 757.28 | 762.62 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 766.20 | 764.34 | 764.29 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 760.00 | 763.97 | 764.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 751.90 | 761.62 | 763.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 757.95 | 753.67 | 757.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 757.20 | 754.37 | 757.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 753.05 | 755.85 | 757.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 715.40 | 722.92 | 727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 722.50 | 722.02 | 726.09 | SL hit (close>ema200) qty=0.50 sl=722.02 alert=retest2 |

### Cycle 190 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 760.00 | 728.81 | 724.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 769.90 | 737.03 | 728.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 756.60 | 759.12 | 745.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 757.25 | 759.12 | 745.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 778.00 | 784.04 | 777.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 778.00 | 784.04 | 777.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 779.00 | 783.03 | 778.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 769.00 | 783.03 | 778.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 772.85 | 780.99 | 777.57 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 771.00 | 775.62 | 775.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 770.00 | 774.49 | 775.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 775.75 | 773.40 | 774.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 789.70 | 776.66 | 775.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 802.45 | 783.68 | 779.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 792.00 | 792.77 | 788.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 774.20 | 792.77 | 788.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 782.05 | 790.62 | 787.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:30:00 | 789.45 | 791.59 | 788.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 780.30 | 788.36 | 788.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 780.30 | 788.36 | 788.79 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 799.35 | 790.23 | 789.25 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 787.70 | 792.98 | 792.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 14:15:00 | 785.55 | 789.24 | 790.93 | Break + close below crossover candle low |

### Cycle 196 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 804.30 | 792.20 | 791.98 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 784.50 | 795.20 | 796.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 769.65 | 783.98 | 788.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 775.65 | 775.35 | 779.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 775.65 | 775.35 | 779.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 772.05 | 774.69 | 777.42 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 788.95 | 779.99 | 778.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 790.00 | 781.99 | 779.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 789.05 | 794.06 | 790.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 787.70 | 792.79 | 789.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:15:00 | 786.60 | 792.79 | 789.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 789.00 | 790.21 | 789.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 789.00 | 790.21 | 789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 788.55 | 789.88 | 789.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 785.00 | 789.88 | 789.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 782.50 | 788.40 | 788.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 777.35 | 783.51 | 786.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 788.80 | 781.87 | 783.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 787.90 | 783.08 | 783.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 782.00 | 783.08 | 783.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 786.15 | 783.75 | 783.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 742.90 | 752.35 | 760.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 746.84 | 752.35 | 760.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 731.50 | 728.21 | 734.68 | SL hit (close>ema200) qty=0.50 sl=728.21 alert=retest2 |

### Cycle 200 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 733.20 | 730.01 | 729.67 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 724.50 | 729.57 | 729.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 722.30 | 728.12 | 729.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 729.50 | 726.68 | 727.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 733.55 | 728.05 | 728.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 733.55 | 728.05 | 728.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 732.30 | 728.90 | 728.75 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 725.95 | 728.87 | 729.16 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 12:15:00 | 730.35 | 728.30 | 728.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 13:15:00 | 737.75 | 730.19 | 729.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 736.80 | 739.04 | 733.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 736.80 | 739.04 | 733.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 734.85 | 738.20 | 734.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 734.85 | 738.20 | 734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 734.80 | 737.52 | 734.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 734.80 | 737.52 | 734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 731.25 | 736.27 | 733.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 731.25 | 736.27 | 733.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 733.30 | 735.67 | 733.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 737.00 | 734.70 | 733.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 735.00 | 735.47 | 734.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 737.05 | 735.26 | 734.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 735.95 | 735.00 | 734.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 734.95 | 734.99 | 734.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 733.30 | 734.28 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 733.30 | 734.28 | 734.38 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 736.95 | 734.41 | 734.28 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 730.00 | 733.53 | 733.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 727.00 | 730.71 | 732.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 726.30 | 728.98 | 729.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 726.70 | 728.51 | 729.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 734.60 | 730.33 | 730.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 734.60 | 730.33 | 730.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 736.45 | 732.44 | 731.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 732.10 | 732.89 | 732.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 727.65 | 731.84 | 731.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 727.65 | 731.84 | 731.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 729.80 | 731.43 | 731.51 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 733.80 | 731.91 | 731.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 735.00 | 732.52 | 732.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 731.25 | 732.27 | 731.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 730.00 | 731.82 | 731.77 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 730.00 | 731.45 | 731.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 728.25 | 730.54 | 731.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 732.50 | 728.87 | 729.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 731.25 | 729.34 | 729.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 731.60 | 729.34 | 729.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 729.25 | 729.27 | 729.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 731.15 | 729.27 | 729.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 732.30 | 729.88 | 729.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 727.35 | 729.53 | 729.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 726.70 | 729.53 | 729.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 726.55 | 714.20 | 715.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 727.10 | 718.93 | 717.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 727.10 | 718.93 | 717.93 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 712.25 | 717.06 | 717.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 10:15:00 | 709.80 | 714.63 | 716.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 709.80 | 709.10 | 711.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:00:00 | 709.80 | 709.10 | 711.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 715.65 | 710.41 | 712.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 715.65 | 710.41 | 712.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 713.70 | 711.07 | 712.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 710.75 | 711.32 | 712.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 712.60 | 711.92 | 712.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 714.40 | 712.48 | 712.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 714.40 | 712.48 | 712.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 715.90 | 713.16 | 712.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 713.20 | 718.92 | 717.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 717.35 | 718.61 | 717.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 712.85 | 718.61 | 717.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 718.05 | 718.50 | 717.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 720.00 | 718.81 | 717.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 720.20 | 718.87 | 717.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 713.65 | 719.90 | 719.69 | SL hit (close<static) qty=1.00 sl=715.40 alert=retest2 |

### Cycle 215 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 716.30 | 719.18 | 719.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 711.90 | 716.38 | 717.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 709.70 | 708.10 | 710.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 709.50 | 707.69 | 709.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 709.50 | 707.69 | 709.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 707.65 | 707.68 | 709.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 708.85 | 707.68 | 709.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 712.30 | 708.61 | 709.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 712.30 | 708.61 | 709.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 712.45 | 709.38 | 710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 712.45 | 709.38 | 710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 710.60 | 710.15 | 710.29 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 713.75 | 710.99 | 710.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 718.50 | 713.51 | 712.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 708.70 | 713.27 | 712.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 709.50 | 712.51 | 712.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 708.35 | 712.51 | 712.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 711.40 | 712.15 | 712.18 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 715.50 | 712.82 | 712.48 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 705.40 | 712.15 | 712.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 701.40 | 710.00 | 711.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 701.05 | 699.23 | 703.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 701.05 | 699.23 | 703.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 702.80 | 699.80 | 702.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 702.05 | 699.80 | 702.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 711.75 | 702.19 | 703.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 711.75 | 702.19 | 703.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 712.00 | 704.15 | 704.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 716.95 | 706.71 | 705.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 711.55 | 714.21 | 711.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 710.15 | 713.40 | 711.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 705.65 | 713.40 | 711.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 710.10 | 712.74 | 711.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 710.10 | 712.74 | 711.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 705.00 | 709.51 | 710.03 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 724.80 | 708.81 | 708.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 730.30 | 719.66 | 714.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 729.40 | 722.79 | 719.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:30:00 | 729.30 | 725.04 | 721.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 15:15:00 | 730.65 | 725.70 | 721.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 09:30:00 | 730.80 | 727.38 | 723.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 744.80 | 738.80 | 732.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 746.90 | 738.80 | 732.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 745.05 | 743.67 | 737.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 746.00 | 745.11 | 739.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:45:00 | 745.55 | 742.84 | 740.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 741.65 | 742.73 | 741.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 741.65 | 742.73 | 741.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 743.75 | 742.93 | 741.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 744.70 | 743.90 | 742.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 748.80 | 747.02 | 744.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 742.60 | 746.41 | 746.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 742.60 | 746.41 | 746.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 740.50 | 745.15 | 745.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 746.10 | 744.07 | 745.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 745.40 | 744.34 | 745.20 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 750.15 | 745.96 | 745.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 09:15:00 | 754.85 | 749.34 | 747.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 747.70 | 749.12 | 747.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 756.05 | 750.50 | 748.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:30:00 | 757.85 | 753.63 | 750.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 757.95 | 755.73 | 752.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 749.40 | 754.59 | 754.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 749.40 | 754.59 | 754.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 749.00 | 751.55 | 752.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 722.00 | 715.35 | 721.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 721.80 | 716.64 | 721.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 721.80 | 716.64 | 721.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 725.20 | 718.35 | 722.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 725.20 | 718.35 | 722.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 723.50 | 719.38 | 722.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:45:00 | 723.00 | 719.38 | 722.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 722.90 | 720.08 | 722.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 716.65 | 719.58 | 721.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-20 09:15:00 | 644.99 | 693.99 | 696.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 668.00 | 661.21 | 660.64 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 655.30 | 659.44 | 659.90 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 664.15 | 660.83 | 660.42 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 658.00 | 660.31 | 660.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 654.40 | 659.13 | 659.78 | Break + close below crossover candle low |

### Cycle 230 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 668.80 | 661.07 | 660.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 673.10 | 663.47 | 661.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 660.50 | 664.54 | 663.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 660.50 | 663.73 | 662.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 647.15 | 663.73 | 662.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 653.40 | 661.66 | 661.93 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 664.00 | 659.10 | 659.02 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 657.80 | 659.93 | 659.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 652.05 | 657.56 | 658.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 647.95 | 655.83 | 657.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 660.35 | 652.23 | 653.06 | SL hit (close>static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 661.95 | 654.17 | 653.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 665.60 | 656.46 | 654.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 664.05 | 670.19 | 665.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 672.25 | 670.60 | 665.98 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 659.15 | 664.36 | 664.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 654.60 | 661.07 | 663.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 640.55 | 636.99 | 643.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 640.55 | 636.99 | 643.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 642.00 | 637.97 | 641.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 642.00 | 637.97 | 641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 641.75 | 638.72 | 641.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 641.75 | 638.72 | 641.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 639.90 | 638.96 | 641.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:15:00 | 644.00 | 638.96 | 641.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 644.00 | 639.97 | 641.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:30:00 | 638.00 | 640.69 | 641.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 638.45 | 639.76 | 641.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 637.00 | 639.81 | 641.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 638.70 | 638.95 | 640.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 640.50 | 639.26 | 640.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 640.50 | 639.26 | 640.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 641.35 | 639.68 | 640.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 641.85 | 639.68 | 640.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 639.80 | 639.70 | 640.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 639.80 | 639.70 | 640.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 641.50 | 640.06 | 640.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 637.45 | 640.06 | 640.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 633.00 | 629.86 | 629.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 633.00 | 629.86 | 629.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 637.35 | 631.36 | 630.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA400 retest candle locked (from upside) |

### Cycle 237 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 629.95 | 631.66 | 631.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 624.70 | 630.09 | 631.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 629.95 | 628.80 | 630.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 629.95 | 628.80 | 630.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 630.00 | 629.04 | 630.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 630.00 | 629.04 | 630.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 630.80 | 629.39 | 630.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 630.65 | 629.39 | 630.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 630.00 | 629.51 | 630.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 632.00 | 629.51 | 630.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 633.05 | 630.22 | 630.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 632.00 | 630.22 | 630.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 238 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 632.55 | 630.69 | 630.60 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 14:15:00 | 630.20 | 630.60 | 630.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 09:15:00 | 618.40 | 628.06 | 629.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 608.05 | 607.03 | 613.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 09:15:00 | 613.90 | 607.03 | 613.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 615.00 | 608.62 | 613.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 616.00 | 608.62 | 613.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 614.75 | 609.85 | 613.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 616.50 | 609.85 | 613.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 616.90 | 614.38 | 614.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 616.95 | 614.38 | 614.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 615.35 | 614.57 | 614.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 613.25 | 614.57 | 614.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 618.95 | 615.45 | 615.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 618.95 | 615.45 | 615.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 633.10 | 618.98 | 616.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 620.20 | 621.36 | 618.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 620.20 | 621.36 | 618.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 615.30 | 620.15 | 618.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 612.05 | 620.15 | 618.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 612.90 | 618.70 | 617.85 | EMA400 retest candle locked (from upside) |

### Cycle 241 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 611.00 | 617.08 | 617.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 610.25 | 614.88 | 616.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 609.45 | 604.62 | 608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 618.00 | 607.30 | 609.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 618.00 | 607.30 | 609.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 623.00 | 610.44 | 610.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 623.00 | 610.44 | 610.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 242 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 623.50 | 613.05 | 611.94 | EMA200 above EMA400 |

### Cycle 243 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 611.05 | 614.94 | 615.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 606.50 | 609.92 | 612.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 611.40 | 609.29 | 611.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 622.00 | 611.83 | 612.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 622.00 | 611.83 | 612.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 244 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 632.00 | 615.86 | 613.93 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 607.40 | 613.86 | 613.97 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 620.15 | 613.08 | 612.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 626.40 | 615.89 | 614.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA400 retest candle locked (from upside) |

### Cycle 247 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 598.40 | 615.15 | 617.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 596.35 | 611.39 | 615.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 623.75 | 610.25 | 612.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 628.00 | 613.80 | 614.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 626.25 | 613.80 | 614.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 248 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 623.95 | 615.83 | 615.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 635.15 | 624.62 | 622.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 628.60 | 628.81 | 626.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:30:00 | 628.15 | 628.81 | 626.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 626.95 | 628.44 | 626.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 634.15 | 631.39 | 627.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 630.80 | 637.66 | 638.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 249 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 630.80 | 637.66 | 638.35 | EMA200 below EMA400 |

### Cycle 250 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 644.25 | 639.10 | 638.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 647.20 | 642.01 | 640.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 657.20 | 660.50 | 654.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 657.20 | 660.50 | 654.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 654.95 | 659.39 | 654.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 654.95 | 659.39 | 654.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 651.80 | 657.87 | 654.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 656.80 | 657.66 | 654.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 660.70 | 657.66 | 654.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:45:00 | 657.85 | 657.34 | 655.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 663.15 | 677.82 | 678.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 251 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 663.15 | 677.82 | 678.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 661.65 | 674.59 | 676.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 673.10 | 669.93 | 673.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 672.50 | 670.44 | 673.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 672.50 | 670.44 | 673.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 671.40 | 670.63 | 673.24 | EMA400 retest candle locked (from downside) |

### Cycle 252 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 680.70 | 675.11 | 674.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 681.65 | 677.20 | 675.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 672.40 | 677.82 | 676.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 678.60 | 677.98 | 677.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 681.85 | 677.98 | 677.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 682.80 | 679.63 | 677.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 675.25 | 680.09 | 680.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 253 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 675.25 | 680.09 | 680.58 | EMA200 below EMA400 |

### Cycle 254 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 684.40 | 680.13 | 680.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 687.30 | 682.41 | 681.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 680.75 | 683.30 | 682.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 677.55 | 682.15 | 681.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 676.20 | 682.15 | 681.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 680.10 | 681.55 | 681.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 680.00 | 681.55 | 681.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 255 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 680.00 | 681.24 | 681.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 677.00 | 680.24 | 680.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 679.30 | 679.91 | 680.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 681.10 | 680.15 | 680.58 | EMA400 retest candle locked (from downside) |

### Cycle 256 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 682.70 | 681.15 | 680.99 | EMA200 above EMA400 |

### Cycle 257 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 677.30 | 680.38 | 680.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 675.45 | 679.40 | 680.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-19 13:15:00 | 383.40 | 2023-05-24 14:15:00 | 382.85 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-05-19 13:45:00 | 382.95 | 2023-05-24 14:15:00 | 382.85 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-05-19 14:30:00 | 384.30 | 2023-05-25 14:15:00 | 381.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-05-23 09:15:00 | 387.20 | 2023-05-25 14:15:00 | 381.40 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-05-24 09:30:00 | 391.30 | 2023-05-25 14:15:00 | 381.40 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2023-05-24 11:00:00 | 390.05 | 2023-05-25 14:15:00 | 381.40 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2023-05-25 10:45:00 | 390.95 | 2023-05-25 14:15:00 | 381.40 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2023-05-29 14:15:00 | 378.20 | 2023-05-31 15:15:00 | 383.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-05-30 09:30:00 | 378.50 | 2023-06-01 10:15:00 | 384.95 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-05-30 10:45:00 | 378.50 | 2023-06-01 10:15:00 | 384.95 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-05-30 12:45:00 | 378.50 | 2023-06-01 10:15:00 | 384.95 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-05-31 12:15:00 | 374.25 | 2023-06-01 10:15:00 | 384.95 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2023-06-15 12:15:00 | 413.00 | 2023-06-16 15:15:00 | 407.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-06-15 13:30:00 | 413.10 | 2023-06-16 15:15:00 | 407.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-06-15 14:45:00 | 413.00 | 2023-06-16 15:15:00 | 407.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-06-16 11:30:00 | 413.20 | 2023-06-16 15:15:00 | 407.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-06-19 10:00:00 | 412.50 | 2023-06-21 14:15:00 | 410.45 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-06-26 14:00:00 | 423.35 | 2023-06-26 14:15:00 | 418.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-07-11 10:45:00 | 405.00 | 2023-07-12 09:15:00 | 414.55 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2023-07-11 12:00:00 | 404.90 | 2023-07-12 09:15:00 | 414.55 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2023-07-14 09:15:00 | 414.20 | 2023-07-19 10:15:00 | 455.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-09 09:15:00 | 487.65 | 2023-08-11 12:15:00 | 486.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-08-09 13:00:00 | 494.80 | 2023-08-11 12:15:00 | 486.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2023-08-10 14:30:00 | 487.50 | 2023-08-11 12:15:00 | 486.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-08-11 12:00:00 | 486.20 | 2023-08-11 12:15:00 | 486.05 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2023-08-16 14:45:00 | 479.75 | 2023-08-22 09:15:00 | 494.00 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2023-08-16 15:15:00 | 479.50 | 2023-08-22 09:15:00 | 494.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2023-08-18 12:15:00 | 479.75 | 2023-08-22 09:15:00 | 494.00 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2023-08-18 12:45:00 | 479.55 | 2023-08-22 09:15:00 | 494.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2023-09-08 11:15:00 | 517.00 | 2023-09-12 09:15:00 | 506.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2023-09-18 14:30:00 | 492.50 | 2023-09-25 14:15:00 | 492.85 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-09-20 09:15:00 | 492.15 | 2023-09-25 14:15:00 | 492.85 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-10-05 10:30:00 | 482.00 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-10-05 13:15:00 | 482.10 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-10-06 09:15:00 | 482.35 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-06 10:30:00 | 481.15 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-10-10 11:00:00 | 472.90 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2023-10-10 11:30:00 | 472.85 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-10-10 12:00:00 | 472.40 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2023-10-10 14:30:00 | 472.10 | 2023-10-11 10:15:00 | 485.50 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2023-10-23 09:15:00 | 491.05 | 2023-10-23 09:15:00 | 481.45 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-11-03 09:15:00 | 478.25 | 2023-11-09 09:15:00 | 526.08 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-23 14:30:00 | 538.90 | 2023-11-24 09:15:00 | 547.75 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-11-23 15:00:00 | 538.95 | 2023-11-24 09:15:00 | 547.75 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest1 | 2023-12-11 09:15:00 | 523.20 | 2023-12-12 10:15:00 | 530.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest1 | 2023-12-11 15:15:00 | 526.00 | 2023-12-12 10:15:00 | 530.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-12-19 09:15:00 | 538.80 | 2023-12-20 13:15:00 | 535.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-12-20 12:00:00 | 540.10 | 2023-12-20 13:15:00 | 535.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-01-02 10:15:00 | 545.75 | 2024-01-02 12:15:00 | 554.45 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-01-09 13:45:00 | 576.50 | 2024-01-11 09:15:00 | 569.45 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-01-10 11:45:00 | 573.60 | 2024-01-11 09:15:00 | 569.45 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-01-10 14:45:00 | 574.75 | 2024-01-11 09:15:00 | 569.45 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-01-31 09:15:00 | 595.20 | 2024-02-01 11:15:00 | 579.50 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-02-15 15:00:00 | 539.65 | 2024-02-15 15:15:00 | 539.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-02-28 09:15:00 | 522.55 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2024-02-28 14:45:00 | 522.50 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2024-03-01 09:15:00 | 520.55 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-03-02 12:00:00 | 521.50 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-03-04 10:30:00 | 517.55 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-03-06 09:30:00 | 513.90 | 2024-03-06 15:15:00 | 519.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest1 | 2024-03-14 09:15:00 | 498.05 | 2024-03-20 11:15:00 | 490.65 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2024-03-19 10:30:00 | 483.30 | 2024-03-20 15:15:00 | 490.75 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-03-20 09:45:00 | 482.25 | 2024-03-20 15:15:00 | 490.75 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-04-09 09:15:00 | 544.00 | 2024-04-15 09:15:00 | 527.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-04-19 14:15:00 | 524.00 | 2024-04-24 11:15:00 | 530.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-04-24 11:00:00 | 523.00 | 2024-04-24 11:15:00 | 530.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-04-29 09:15:00 | 529.70 | 2024-05-03 15:15:00 | 530.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-04-30 09:15:00 | 532.75 | 2024-05-03 15:15:00 | 530.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-05-02 13:30:00 | 531.00 | 2024-05-03 15:15:00 | 530.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-05-03 14:45:00 | 531.80 | 2024-05-03 15:15:00 | 530.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-05-09 13:00:00 | 514.10 | 2024-05-10 11:15:00 | 530.70 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-06-04 10:15:00 | 575.95 | 2024-06-04 11:15:00 | 551.05 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2024-06-20 09:15:00 | 732.20 | 2024-06-20 13:15:00 | 718.70 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-06-24 11:15:00 | 722.80 | 2024-06-24 11:15:00 | 721.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-06-25 09:15:00 | 721.45 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-06-25 11:15:00 | 720.35 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-06-26 12:00:00 | 721.55 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-07-05 10:30:00 | 714.70 | 2024-07-11 11:15:00 | 702.80 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-07-05 11:00:00 | 714.85 | 2024-07-11 11:15:00 | 702.80 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-07-18 09:15:00 | 718.55 | 2024-07-18 09:15:00 | 703.25 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-18 15:15:00 | 700.00 | 2024-07-22 11:15:00 | 706.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-07-19 09:45:00 | 698.35 | 2024-07-22 11:15:00 | 706.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-07-25 12:15:00 | 712.45 | 2024-07-26 09:15:00 | 703.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-25 14:00:00 | 712.65 | 2024-07-26 09:15:00 | 703.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-08-23 09:45:00 | 847.10 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-08-23 10:15:00 | 848.05 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-08-23 10:45:00 | 848.00 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-09-10 12:00:00 | 857.00 | 2024-09-10 14:15:00 | 862.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-03 13:00:00 | 918.90 | 2024-10-07 09:15:00 | 872.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 13:45:00 | 919.15 | 2024-10-07 09:15:00 | 873.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 14:45:00 | 921.10 | 2024-10-07 09:15:00 | 875.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 13:00:00 | 918.90 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2024-10-03 13:45:00 | 919.15 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.18% |
| SELL | retest2 | 2024-10-03 14:45:00 | 921.10 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.38% |
| BUY | retest2 | 2024-10-18 12:30:00 | 918.55 | 2024-10-21 14:15:00 | 885.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-10-24 14:15:00 | 840.00 | 2024-10-25 09:15:00 | 870.25 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-11-12 10:30:00 | 860.80 | 2024-11-14 09:15:00 | 817.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:30:00 | 860.80 | 2024-11-18 09:15:00 | 774.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-06 11:30:00 | 832.50 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-12-09 09:30:00 | 825.05 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-12-09 10:15:00 | 824.50 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-12-18 09:15:00 | 797.50 | 2024-12-19 10:15:00 | 757.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:15:00 | 797.50 | 2024-12-20 09:15:00 | 777.05 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2024-12-18 09:45:00 | 789.35 | 2024-12-23 14:15:00 | 785.50 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-01-07 12:00:00 | 732.05 | 2025-01-13 09:15:00 | 695.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 12:00:00 | 732.05 | 2025-01-14 15:15:00 | 690.00 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-01-22 09:15:00 | 684.65 | 2025-01-27 10:15:00 | 658.07 | PARTIAL | 0.50 | 3.88% |
| SELL | retest2 | 2025-01-23 13:00:00 | 692.70 | 2025-01-27 10:15:00 | 657.50 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-01-23 15:00:00 | 692.10 | 2025-01-27 10:15:00 | 657.21 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-01-24 09:15:00 | 691.80 | 2025-01-27 14:15:00 | 650.42 | PARTIAL | 0.50 | 5.98% |
| SELL | retest2 | 2025-01-24 12:00:00 | 685.65 | 2025-01-27 14:15:00 | 651.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 667.30 | 2025-01-28 10:15:00 | 633.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 684.65 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-01-23 13:00:00 | 692.70 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.16% |
| SELL | retest2 | 2025-01-23 15:00:00 | 692.10 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.08% |
| SELL | retest2 | 2025-01-24 09:15:00 | 691.80 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-01-24 12:00:00 | 685.65 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2025-01-27 09:15:00 | 667.30 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2025-02-04 15:15:00 | 722.00 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.00% |
| BUY | retest2 | 2025-02-05 12:00:00 | 723.20 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.15% |
| BUY | retest2 | 2025-02-06 10:15:00 | 722.25 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest1 | 2025-02-13 10:15:00 | 600.50 | 2025-02-17 09:15:00 | 570.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 10:45:00 | 600.95 | 2025-02-17 09:15:00 | 570.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 11:30:00 | 600.70 | 2025-02-17 09:15:00 | 570.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 12:00:00 | 600.75 | 2025-02-17 09:15:00 | 570.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 10:15:00 | 600.50 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest1 | 2025-02-13 10:45:00 | 600.95 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest1 | 2025-02-13 11:30:00 | 600.70 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest1 | 2025-02-13 12:00:00 | 600.75 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.95% |
| BUY | retest2 | 2025-02-25 09:15:00 | 649.65 | 2025-02-27 09:15:00 | 632.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-02-28 09:15:00 | 616.70 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2025-02-28 12:30:00 | 618.05 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-02-28 14:15:00 | 619.50 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-03-03 09:15:00 | 615.85 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-04-04 11:00:00 | 689.60 | 2025-04-04 13:15:00 | 669.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-04-17 11:45:00 | 720.15 | 2025-04-25 13:15:00 | 731.25 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-04-17 12:45:00 | 716.45 | 2025-04-25 13:15:00 | 731.25 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-05-06 11:00:00 | 688.20 | 2025-05-08 09:15:00 | 706.10 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-07 14:30:00 | 688.70 | 2025-05-08 09:15:00 | 706.10 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-05-16 14:45:00 | 710.60 | 2025-05-21 13:15:00 | 711.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-17 10:45:00 | 737.65 | 2025-06-19 13:15:00 | 700.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:45:00 | 737.65 | 2025-06-20 09:15:00 | 711.75 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-07-01 10:15:00 | 742.00 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-07-01 13:45:00 | 741.90 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-01 14:30:00 | 742.60 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-01 15:00:00 | 741.25 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-07-10 09:15:00 | 736.05 | 2025-07-10 10:15:00 | 732.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-21 11:15:00 | 754.35 | 2025-07-28 09:15:00 | 755.85 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-21 15:15:00 | 753.00 | 2025-07-28 09:15:00 | 755.85 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-08-01 15:15:00 | 753.05 | 2025-08-11 09:15:00 | 715.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 15:15:00 | 753.05 | 2025-08-11 11:15:00 | 722.50 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2025-08-28 10:30:00 | 789.45 | 2025-08-29 10:15:00 | 780.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-22 09:15:00 | 782.00 | 2025-09-25 14:15:00 | 742.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:30:00 | 786.15 | 2025-09-25 14:15:00 | 746.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 782.00 | 2025-09-30 10:15:00 | 731.50 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2025-09-22 11:30:00 | 786.15 | 2025-09-30 10:15:00 | 731.50 | STOP_HIT | 0.50 | 6.95% |
| BUY | retest2 | 2025-10-16 09:15:00 | 737.00 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-16 14:30:00 | 735.00 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-17 09:15:00 | 737.05 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-17 10:15:00 | 735.95 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-28 10:15:00 | 726.30 | 2025-10-28 13:15:00 | 734.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 12:00:00 | 726.70 | 2025-10-28 13:15:00 | 734.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-06 09:30:00 | 727.35 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-11-06 10:15:00 | 726.70 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-11 12:00:00 | 726.55 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-11-14 14:45:00 | 710.75 | 2025-11-17 14:15:00 | 714.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-17 13:15:00 | 712.60 | 2025-11-17 14:15:00 | 714.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-19 12:45:00 | 720.00 | 2025-11-21 10:15:00 | 713.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-19 15:00:00 | 720.20 | 2025-11-21 10:15:00 | 713.65 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-12 10:45:00 | 729.40 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2025-12-12 13:30:00 | 729.30 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-12-12 15:15:00 | 730.65 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-12-15 09:30:00 | 730.80 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-12-16 10:15:00 | 746.90 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-16 14:00:00 | 745.05 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-12-17 09:45:00 | 746.00 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-18 10:45:00 | 745.55 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-19 09:30:00 | 744.70 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-12-22 09:45:00 | 748.80 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-29 13:30:00 | 757.85 | 2025-12-31 12:15:00 | 749.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-30 10:00:00 | 757.95 | 2025-12-31 12:15:00 | 749.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-08 10:45:00 | 716.65 | 2026-01-20 09:15:00 | 644.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-06 09:15:00 | 647.95 | 2026-02-09 10:15:00 | 660.35 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-02-18 11:30:00 | 638.00 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-02-18 14:45:00 | 638.45 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2026-02-19 09:15:00 | 637.00 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2026-02-19 12:15:00 | 638.70 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2026-02-20 09:15:00 | 637.45 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-03-11 10:15:00 | 613.25 | 2026-03-11 10:15:00 | 618.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-04-08 09:30:00 | 634.15 | 2026-04-13 10:15:00 | 630.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-04-21 10:00:00 | 656.80 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2026-04-21 10:30:00 | 660.70 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2026-04-21 13:45:00 | 657.85 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2026-04-29 09:15:00 | 681.85 | 2026-04-30 12:15:00 | 675.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-29 09:45:00 | 682.80 | 2026-04-30 12:15:00 | 675.25 | STOP_HIT | 1.00 | -1.11% |
