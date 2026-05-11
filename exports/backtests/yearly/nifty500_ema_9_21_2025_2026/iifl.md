# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 460.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 22 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 65 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 47
- **Target hits / Stop hits / Partials:** 6 / 62 / 4
- **Avg / median % per leg:** 0.39% / -0.80%
- **Sum % (uncompounded):** 27.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 16 | 44.4% | 5 | 31 | 0 | 1.15% | 41.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.44% | -4.3% |
| BUY @ 3rd Alert (retest2) | 33 | 16 | 48.5% | 5 | 28 | 0 | 1.38% | 45.7% |
| SELL (all) | 36 | 9 | 25.0% | 1 | 31 | 4 | -0.38% | -13.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 9 | 25.0% | 1 | 31 | 4 | -0.38% | -13.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.44% | -4.3% |
| retest2 (combined) | 69 | 25 | 36.2% | 6 | 59 | 4 | 0.47% | 32.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 407.65 | 410.67 | 410.78 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 412.70 | 410.99 | 410.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 409.05 | 410.61 | 410.70 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 413.50 | 411.18 | 410.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 414.85 | 412.62 | 411.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 412.40 | 412.58 | 411.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 412.40 | 412.58 | 411.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 413.55 | 412.77 | 411.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 412.75 | 412.77 | 411.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 413.65 | 412.95 | 412.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 412.90 | 412.95 | 412.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 412.85 | 412.93 | 412.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 412.55 | 412.93 | 412.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 414.20 | 413.18 | 412.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 412.45 | 413.18 | 412.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 415.65 | 413.77 | 412.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 424.80 | 414.27 | 413.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 418.70 | 416.48 | 414.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 418.00 | 416.59 | 415.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 419.80 | 417.41 | 415.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 417.20 | 418.34 | 416.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:45:00 | 417.15 | 418.34 | 416.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 417.60 | 418.19 | 416.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:45:00 | 418.15 | 418.54 | 417.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 425.60 | 428.11 | 428.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 425.60 | 428.11 | 428.28 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 428.50 | 427.75 | 427.72 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 427.05 | 427.61 | 427.66 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 429.40 | 427.97 | 427.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 432.60 | 429.93 | 428.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 480.20 | 480.69 | 470.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 485.15 | 480.69 | 470.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:45:00 | 483.40 | 484.68 | 478.92 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:30:00 | 483.60 | 483.95 | 479.55 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 477.10 | 482.57 | 479.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 477.10 | 482.57 | 479.69 | SL hit (close<ema400) qty=1.00 sl=479.69 alert=retest1 |

### Cycle 9 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 470.25 | 478.27 | 478.28 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 482.65 | 475.33 | 475.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 485.20 | 477.30 | 476.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 486.35 | 486.96 | 483.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 489.05 | 486.96 | 483.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 485.40 | 486.65 | 483.22 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 476.85 | 482.88 | 483.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 472.70 | 480.84 | 482.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 478.20 | 477.00 | 479.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 478.20 | 477.00 | 479.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 477.00 | 477.00 | 479.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 479.10 | 477.00 | 479.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 479.05 | 476.64 | 478.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 479.05 | 476.64 | 478.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 479.20 | 477.15 | 478.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 479.25 | 477.15 | 478.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 481.15 | 477.95 | 478.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 482.25 | 477.95 | 478.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 481.80 | 478.72 | 478.71 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 475.80 | 478.41 | 478.60 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 480.05 | 478.70 | 478.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 13:15:00 | 484.00 | 480.11 | 479.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 478.90 | 480.64 | 479.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 475.50 | 479.61 | 479.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 475.50 | 479.61 | 479.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 477.00 | 479.09 | 479.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 470.10 | 473.67 | 475.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 476.50 | 473.26 | 475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 473.80 | 473.37 | 475.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 474.50 | 473.37 | 475.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 474.00 | 472.93 | 474.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 473.15 | 472.93 | 474.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 470.10 | 472.36 | 473.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 467.05 | 472.09 | 473.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 468.25 | 470.05 | 472.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 466.35 | 468.41 | 471.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 471.75 | 466.52 | 466.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 471.75 | 466.52 | 466.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 488.80 | 473.42 | 469.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 515.45 | 516.09 | 508.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 515.45 | 516.09 | 508.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 524.60 | 526.19 | 524.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 524.00 | 526.19 | 524.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 527.40 | 526.43 | 524.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 529.00 | 526.62 | 524.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 530.65 | 526.63 | 525.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 529.05 | 532.84 | 531.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 530.00 | 531.45 | 531.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 530.00 | 531.45 | 531.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 525.00 | 529.93 | 530.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 530.45 | 529.43 | 530.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 529.40 | 529.42 | 530.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 528.35 | 529.42 | 530.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 530.90 | 529.72 | 530.32 | SL hit (close>static) qty=1.00 sl=530.75 alert=retest2 |

### Cycle 18 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 536.50 | 531.71 | 531.16 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 530.00 | 530.83 | 530.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 517.40 | 527.84 | 529.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 506.40 | 504.04 | 510.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 506.40 | 504.04 | 510.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 513.60 | 505.95 | 510.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 513.60 | 505.95 | 510.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 518.05 | 508.37 | 511.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 513.85 | 508.37 | 511.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 441.15 | 440.67 | 447.42 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 453.95 | 449.69 | 449.23 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 446.00 | 449.60 | 449.72 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 450.00 | 449.23 | 449.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 453.20 | 450.03 | 449.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 450.00 | 450.65 | 450.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 446.85 | 449.89 | 449.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 450.30 | 449.89 | 449.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 448.00 | 449.51 | 449.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 448.00 | 449.51 | 449.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 446.80 | 448.97 | 449.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 453.95 | 446.57 | 447.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 447.00 | 446.66 | 447.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 446.75 | 446.66 | 447.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:00:00 | 445.60 | 446.42 | 447.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:00:00 | 446.70 | 445.58 | 446.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 446.80 | 445.83 | 446.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 450.00 | 446.67 | 446.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:00:00 | 450.00 | 446.67 | 446.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-18 14:15:00 | 449.15 | 447.16 | 446.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 449.15 | 447.16 | 446.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 453.55 | 448.61 | 447.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 467.65 | 469.51 | 464.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 465.85 | 468.37 | 464.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 465.85 | 468.37 | 464.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 465.85 | 468.37 | 464.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 461.85 | 467.06 | 464.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 461.85 | 467.06 | 464.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 463.95 | 466.44 | 464.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 461.95 | 466.44 | 464.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 460.40 | 464.14 | 463.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 461.30 | 464.14 | 463.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 458.35 | 462.43 | 462.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 457.00 | 461.34 | 462.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 457.50 | 456.31 | 458.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 457.50 | 456.31 | 458.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 460.65 | 457.18 | 458.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 460.90 | 457.18 | 458.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 460.70 | 457.88 | 458.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 461.90 | 457.88 | 458.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 456.85 | 457.92 | 458.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 454.45 | 457.92 | 458.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 455.25 | 456.93 | 458.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 15:15:00 | 431.73 | 440.96 | 447.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 15:15:00 | 432.49 | 440.96 | 447.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 439.60 | 438.60 | 445.16 | SL hit (close>ema200) qty=0.50 sl=438.60 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 443.60 | 440.55 | 440.30 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 438.45 | 440.15 | 440.20 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 442.80 | 440.26 | 440.21 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 437.80 | 439.76 | 439.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 432.80 | 436.58 | 438.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 437.20 | 434.54 | 435.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 435.15 | 434.66 | 435.86 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 441.40 | 436.63 | 436.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 444.55 | 438.21 | 437.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 443.55 | 444.29 | 441.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 444.15 | 444.29 | 441.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 441.95 | 443.82 | 441.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 441.10 | 443.82 | 441.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 441.80 | 443.42 | 441.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 441.80 | 443.42 | 441.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 443.45 | 443.42 | 441.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 444.30 | 442.26 | 441.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 443.85 | 442.90 | 441.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 446.45 | 445.88 | 444.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 439.90 | 444.64 | 444.05 | SL hit (close<static) qty=1.00 sl=441.25 alert=retest2 |

### Cycle 31 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 438.55 | 442.84 | 443.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 437.25 | 440.34 | 441.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 447.95 | 440.74 | 439.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 451.75 | 442.94 | 441.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 445.55 | 446.03 | 443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 445.85 | 446.00 | 443.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 448.30 | 445.27 | 444.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 449.20 | 452.03 | 451.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 446.00 | 450.83 | 451.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 446.00 | 450.83 | 451.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 443.80 | 448.48 | 449.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 430.90 | 427.28 | 433.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 430.90 | 427.28 | 433.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 432.95 | 428.42 | 433.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 432.95 | 428.42 | 433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 429.20 | 428.57 | 433.02 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 448.85 | 436.94 | 435.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 450.20 | 443.55 | 439.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 445.00 | 445.67 | 441.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:00:00 | 445.00 | 445.67 | 441.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 485.70 | 490.00 | 487.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 485.15 | 490.00 | 487.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 487.35 | 489.47 | 487.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:30:00 | 488.65 | 489.24 | 487.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 488.45 | 489.23 | 487.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 482.90 | 488.17 | 488.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 482.90 | 488.17 | 488.19 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 495.90 | 489.18 | 488.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 492.18 | 489.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 500.20 | 501.37 | 497.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:30:00 | 500.05 | 501.37 | 497.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 499.55 | 500.81 | 498.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 501.75 | 500.81 | 498.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 495.50 | 499.74 | 497.81 | SL hit (close<static) qty=1.00 sl=498.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 491.75 | 496.95 | 497.21 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 501.85 | 497.74 | 497.48 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 491.75 | 497.37 | 497.59 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 504.30 | 496.01 | 495.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 511.55 | 503.44 | 499.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 519.65 | 511.63 | 506.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 517.35 | 512.73 | 507.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:00:00 | 517.40 | 513.66 | 508.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 526.45 | 537.38 | 537.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 526.45 | 537.38 | 537.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 524.35 | 533.42 | 535.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 527.25 | 525.38 | 529.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 526.00 | 525.38 | 529.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 525.15 | 525.51 | 528.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 532.00 | 525.51 | 528.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 526.60 | 525.73 | 528.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 527.80 | 525.73 | 528.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 529.80 | 526.54 | 528.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 533.05 | 526.54 | 528.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 535.35 | 528.30 | 529.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 533.90 | 528.30 | 529.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 542.70 | 531.18 | 530.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 544.15 | 538.68 | 534.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 12:15:00 | 539.25 | 539.74 | 536.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 13:00:00 | 539.25 | 539.74 | 536.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 536.40 | 539.53 | 537.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 536.40 | 539.53 | 537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 539.45 | 539.51 | 537.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 535.50 | 539.51 | 537.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 541.05 | 539.57 | 537.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 542.40 | 540.63 | 539.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 15:15:00 | 536.45 | 538.91 | 538.78 | SL hit (close<static) qty=1.00 sl=537.25 alert=retest2 |

### Cycle 43 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 547.00 | 555.85 | 556.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 542.70 | 548.93 | 552.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 548.00 | 546.25 | 549.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 548.00 | 546.25 | 549.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 546.50 | 544.58 | 547.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 544.40 | 544.58 | 547.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 548.45 | 545.45 | 547.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 549.75 | 545.45 | 547.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 544.40 | 545.24 | 547.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 543.55 | 545.24 | 547.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 554.25 | 547.38 | 546.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 554.25 | 547.38 | 546.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 557.05 | 549.32 | 547.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 566.95 | 568.01 | 562.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 567.60 | 568.01 | 562.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 582.80 | 584.41 | 580.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 581.70 | 584.41 | 580.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 581.45 | 583.54 | 580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 579.70 | 583.54 | 580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 580.95 | 583.02 | 580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 576.20 | 583.02 | 580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 572.85 | 580.99 | 580.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 572.85 | 580.99 | 580.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 570.00 | 578.79 | 579.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 566.45 | 571.93 | 574.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 568.75 | 567.59 | 571.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 568.75 | 567.59 | 571.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 569.60 | 568.00 | 570.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 569.60 | 568.00 | 570.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 558.80 | 566.22 | 569.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 556.30 | 566.22 | 569.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 557.20 | 557.40 | 561.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 574.40 | 565.31 | 564.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 574.40 | 565.31 | 564.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 578.50 | 572.12 | 569.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 571.00 | 574.31 | 571.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 572.65 | 573.98 | 571.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:30:00 | 577.55 | 574.72 | 572.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 565.05 | 572.94 | 572.24 | SL hit (close<static) qty=1.00 sl=570.65 alert=retest2 |

### Cycle 47 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 565.65 | 571.48 | 571.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 564.15 | 566.94 | 568.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 569.25 | 565.33 | 566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 566.15 | 565.49 | 566.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 568.75 | 565.49 | 566.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 561.30 | 564.65 | 566.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 556.65 | 564.65 | 566.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 560.70 | 563.17 | 564.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 560.00 | 561.71 | 563.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 560.70 | 561.61 | 563.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 565.25 | 562.34 | 563.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 565.25 | 562.34 | 563.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 564.40 | 562.75 | 563.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 571.00 | 562.75 | 563.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 572.60 | 564.72 | 564.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 572.60 | 564.72 | 564.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 578.90 | 568.64 | 566.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 569.30 | 571.11 | 568.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 569.30 | 571.11 | 568.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 569.50 | 570.79 | 568.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 572.55 | 570.79 | 568.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 571.80 | 571.50 | 569.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:00:00 | 571.15 | 572.09 | 570.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 572.60 | 572.38 | 570.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 586.10 | 575.22 | 572.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 591.10 | 575.22 | 572.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-02 09:15:00 | 629.80 | 620.29 | 612.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 647.45 | 650.67 | 650.72 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 654.70 | 651.02 | 650.76 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 648.20 | 650.23 | 650.45 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 15:15:00 | 652.30 | 650.58 | 650.56 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 644.00 | 649.26 | 649.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 631.45 | 645.70 | 648.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 644.90 | 644.80 | 647.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 644.90 | 644.80 | 647.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 644.35 | 643.74 | 645.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 637.05 | 641.21 | 644.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 638.55 | 635.51 | 636.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 635.45 | 635.51 | 636.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 639.10 | 636.36 | 636.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 639.25 | 636.94 | 637.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 639.05 | 636.94 | 637.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 645.15 | 638.58 | 637.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 645.15 | 638.58 | 637.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 648.50 | 640.57 | 638.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 631.50 | 643.51 | 640.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 637.65 | 642.33 | 640.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 637.65 | 642.33 | 640.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 640.40 | 641.95 | 640.56 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 631.55 | 639.06 | 639.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 629.15 | 637.08 | 638.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 624.90 | 624.27 | 630.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 621.85 | 622.99 | 628.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 623.80 | 622.99 | 628.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 595.30 | 617.50 | 625.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 591.80 | 617.50 | 625.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-22 14:15:00 | 532.62 | 593.18 | 609.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 11:15:00 | 561.60 | 548.61 | 548.11 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 534.50 | 546.05 | 547.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 516.00 | 530.74 | 538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 494.20 | 490.89 | 504.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 494.20 | 490.89 | 504.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 503.45 | 494.71 | 503.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 494.35 | 493.92 | 502.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 528.00 | 506.93 | 505.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 528.00 | 506.93 | 505.73 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 498.95 | 508.55 | 509.29 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 513.25 | 508.24 | 508.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 516.80 | 511.15 | 509.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 525.70 | 525.95 | 522.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:00:00 | 525.70 | 525.95 | 522.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 521.45 | 524.97 | 523.13 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 501.05 | 519.14 | 521.16 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 520.00 | 514.04 | 513.30 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 500.75 | 511.14 | 512.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 498.00 | 504.70 | 508.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 504.80 | 496.80 | 500.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 498.00 | 497.04 | 500.59 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 515.40 | 503.37 | 502.74 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 501.35 | 503.21 | 503.22 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 507.65 | 503.86 | 503.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 509.40 | 505.86 | 504.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 505.50 | 506.30 | 505.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:15:00 | 503.30 | 506.30 | 505.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 506.00 | 506.24 | 505.18 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 497.45 | 503.31 | 504.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 494.20 | 497.87 | 500.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 498.25 | 497.68 | 500.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 498.25 | 497.68 | 500.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 489.10 | 495.96 | 499.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:30:00 | 498.80 | 495.96 | 499.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 498.10 | 495.82 | 498.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 498.10 | 495.82 | 498.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 503.00 | 497.26 | 498.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 482.45 | 497.26 | 498.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 458.33 | 482.58 | 486.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 473.70 | 472.87 | 478.02 | SL hit (close>ema200) qty=0.50 sl=472.87 alert=retest2 |

### Cycle 68 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 491.15 | 482.30 | 481.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 492.90 | 484.42 | 482.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 484.45 | 485.01 | 482.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 483.25 | 484.66 | 482.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 483.25 | 484.66 | 482.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 481.00 | 483.93 | 482.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 481.00 | 483.93 | 482.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 483.50 | 483.84 | 482.82 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 475.70 | 481.80 | 482.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 475.05 | 480.45 | 481.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 480.65 | 480.49 | 481.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 481.60 | 480.71 | 481.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 477.60 | 480.91 | 481.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:45:00 | 475.40 | 478.66 | 479.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 15:00:00 | 477.30 | 478.39 | 479.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 483.85 | 474.91 | 474.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 483.85 | 474.91 | 474.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 487.00 | 477.33 | 475.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 467.20 | 476.17 | 476.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 447.70 | 469.10 | 472.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 452.30 | 452.19 | 459.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 450.70 | 452.19 | 459.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 453.45 | 452.44 | 458.53 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 467.65 | 461.07 | 460.75 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 449.95 | 460.23 | 460.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 447.85 | 456.02 | 458.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 426.70 | 439.87 | 442.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 429.00 | 432.82 | 436.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 430.75 | 432.82 | 436.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 428.75 | 434.97 | 435.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 441.90 | 432.59 | 433.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 448.45 | 435.77 | 434.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 448.45 | 435.77 | 434.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 453.25 | 448.75 | 445.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 446.60 | 450.02 | 447.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 15:15:00 | 467.95 | 470.44 | 470.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 467.95 | 470.44 | 470.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 447.85 | 465.92 | 468.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 437.25 | 421.38 | 429.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 437.50 | 424.61 | 430.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 435.25 | 424.61 | 430.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 443.50 | 433.62 | 433.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 449.40 | 439.57 | 436.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 445.50 | 445.75 | 441.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 445.50 | 445.75 | 441.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 447.80 | 456.49 | 453.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 447.80 | 456.49 | 453.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 441.50 | 453.49 | 452.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 440.00 | 453.49 | 452.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 441.90 | 451.18 | 451.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 440.05 | 445.27 | 448.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 454.15 | 449.06 | 448.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 457.00 | 450.64 | 449.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 461.55 | 463.35 | 460.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 461.55 | 463.35 | 460.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 460.10 | 462.70 | 460.09 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 14:30:00 | 413.50 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-05-19 13:00:00 | 414.50 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-19 15:15:00 | 415.60 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-05-20 10:30:00 | 414.35 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-05-26 09:15:00 | 424.80 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-26 12:00:00 | 418.70 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-05-26 13:15:00 | 418.00 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-05-27 10:30:00 | 419.80 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-05-28 09:45:00 | 418.15 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest1 | 2025-06-11 09:15:00 | 485.15 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2025-06-12 09:45:00 | 483.40 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-06-12 11:30:00 | 483.60 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-07-01 13:15:00 | 467.05 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-01 15:00:00 | 468.25 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-02 09:30:00 | 466.35 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-17 11:30:00 | 529.00 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-07-18 09:15:00 | 530.65 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-22 12:15:00 | 529.05 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-07-23 13:15:00 | 528.35 | 2025-07-23 13:15:00 | 530.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-13 09:15:00 | 450.30 | 2025-08-13 09:15:00 | 448.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-08-14 11:15:00 | 446.75 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-08-14 13:00:00 | 445.60 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-08-18 11:00:00 | 446.70 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-18 12:45:00 | 446.80 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-26 09:15:00 | 454.45 | 2025-08-28 15:15:00 | 431.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 11:00:00 | 455.25 | 2025-08-28 15:15:00 | 432.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 454.45 | 2025-08-29 10:15:00 | 439.60 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-08-26 11:00:00 | 455.25 | 2025-08-29 10:15:00 | 439.60 | STOP_HIT | 0.50 | 3.44% |
| BUY | retest2 | 2025-09-10 09:15:00 | 444.30 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-10 09:45:00 | 443.85 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-11 10:15:00 | 446.45 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-18 09:30:00 | 448.30 | 2025-09-23 10:15:00 | 446.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-23 10:00:00 | 449.20 | 2025-09-23 10:15:00 | 446.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-13 11:30:00 | 488.65 | 2025-10-14 12:15:00 | 482.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-13 12:30:00 | 488.45 | 2025-10-14 12:15:00 | 482.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-17 09:15:00 | 501.75 | 2025-10-17 09:15:00 | 495.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-29 11:15:00 | 519.65 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-10-29 12:15:00 | 517.35 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-10-29 13:00:00 | 517.40 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-11-13 11:45:00 | 542.40 | 2025-11-13 15:15:00 | 536.45 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-14 09:15:00 | 551.00 | 2025-11-20 10:15:00 | 547.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-24 13:15:00 | 543.55 | 2025-11-25 13:15:00 | 554.25 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-12-08 10:15:00 | 556.30 | 2025-12-10 09:15:00 | 574.40 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-12-09 11:30:00 | 557.20 | 2025-12-10 09:15:00 | 574.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-12-15 13:30:00 | 577.55 | 2025-12-16 09:15:00 | 565.05 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-12-18 14:15:00 | 556.65 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-12-19 11:15:00 | 560.70 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-12-19 12:45:00 | 560.00 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-12-19 13:45:00 | 560.70 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-23 09:15:00 | 572.55 | 2026-01-02 09:15:00 | 629.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 09:45:00 | 571.80 | 2026-01-02 09:15:00 | 628.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 13:00:00 | 571.15 | 2026-01-02 09:15:00 | 628.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 14:45:00 | 572.60 | 2026-01-02 09:15:00 | 629.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 10:15:00 | 591.10 | 2026-01-06 09:15:00 | 650.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 637.05 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-19 09:45:00 | 638.55 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-19 10:15:00 | 635.45 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-19 11:15:00 | 639.10 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-22 10:15:00 | 591.80 | 2026-01-22 14:15:00 | 532.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-22 14:30:00 | 560.60 | 2026-01-22 15:15:00 | 532.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 14:30:00 | 560.60 | 2026-01-27 15:15:00 | 528.15 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2026-02-03 10:45:00 | 494.35 | 2026-02-04 09:15:00 | 528.00 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2026-03-04 09:15:00 | 482.45 | 2026-03-09 09:15:00 | 458.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 482.45 | 2026-03-10 09:15:00 | 473.70 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2026-03-13 09:15:00 | 477.60 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-13 13:45:00 | 475.40 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-13 15:00:00 | 477.30 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-04-02 09:15:00 | 426.70 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-04-06 09:15:00 | 429.00 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2026-04-06 09:45:00 | 430.75 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-07 09:30:00 | 428.75 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-04-13 10:15:00 | 446.60 | 2026-04-22 15:15:00 | 467.95 | STOP_HIT | 1.00 | 4.78% |
