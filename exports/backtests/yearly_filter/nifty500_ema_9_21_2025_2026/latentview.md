# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 314.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 38 |
| ALERT2 | 37 |
| ALERT2_SKIP | 20 |
| ALERT3 | 104 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 57 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 39
- **Target hits / Stop hits / Partials:** 9 / 51 / 9
- **Avg / median % per leg:** 1.68% / -0.22%
- **Sum % (uncompounded):** 115.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 8 | 40.0% | 2 | 18 | 0 | 0.88% | 17.7% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.03% | 3.1% |
| BUY @ 3rd Alert (retest2) | 17 | 5 | 29.4% | 2 | 15 | 0 | 0.86% | 14.6% |
| SELL (all) | 49 | 22 | 44.9% | 7 | 33 | 9 | 2.00% | 98.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 49 | 22 | 44.9% | 7 | 33 | 9 | 2.00% | 98.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.03% | 3.1% |
| retest2 (combined) | 66 | 27 | 40.9% | 9 | 48 | 9 | 1.71% | 112.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 420.00 | 421.14 | 421.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 417.45 | 419.87 | 420.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 410.00 | 409.86 | 413.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 15:15:00 | 409.00 | 408.70 | 410.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 409.00 | 408.70 | 410.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 414.95 | 409.41 | 411.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 411.40 | 409.81 | 411.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:15:00 | 414.55 | 409.81 | 411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 417.30 | 411.31 | 411.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 417.30 | 411.31 | 411.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 416.35 | 412.32 | 412.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 418.85 | 413.62 | 412.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 418.00 | 418.12 | 416.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 414.50 | 418.12 | 416.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 425.85 | 419.66 | 417.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 414.15 | 419.66 | 417.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 418.45 | 419.69 | 418.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 418.45 | 419.69 | 418.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 423.85 | 420.52 | 418.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 417.30 | 420.52 | 418.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 421.50 | 420.67 | 419.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 422.45 | 420.63 | 419.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 417.80 | 419.89 | 419.42 | SL hit (close<static) qty=1.00 sl=419.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 415.20 | 419.19 | 419.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 415.05 | 418.37 | 418.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 416.50 | 416.13 | 417.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 416.20 | 416.14 | 416.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 415.70 | 416.14 | 416.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 416.45 | 416.20 | 416.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 416.45 | 416.20 | 416.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 416.50 | 416.26 | 416.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 416.15 | 416.26 | 416.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 416.05 | 416.22 | 416.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 413.35 | 416.18 | 416.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 414.30 | 415.80 | 416.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 414.45 | 415.76 | 416.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 415.00 | 412.15 | 411.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 415.00 | 412.15 | 411.90 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 412.20 | 412.69 | 412.75 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 421.15 | 414.38 | 413.51 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 410.00 | 413.95 | 414.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 409.25 | 413.01 | 413.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 397.80 | 397.54 | 401.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 397.80 | 397.54 | 401.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 401.55 | 398.34 | 401.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 399.80 | 398.34 | 401.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 399.00 | 398.47 | 401.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 418.40 | 398.47 | 401.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 415.50 | 401.88 | 402.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 427.85 | 401.88 | 402.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 418.40 | 405.18 | 404.11 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 396.90 | 403.92 | 404.82 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 407.60 | 403.09 | 403.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 408.50 | 405.96 | 404.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 405.35 | 406.30 | 405.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 406.85 | 406.41 | 405.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 405.95 | 406.41 | 405.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 405.95 | 406.32 | 405.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 408.85 | 406.32 | 405.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:00:00 | 408.50 | 407.29 | 405.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-03 14:15:00 | 449.74 | 436.26 | 426.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 439.45 | 441.64 | 441.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 436.80 | 440.04 | 440.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 428.70 | 429.30 | 432.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 434.60 | 430.36 | 432.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 434.60 | 430.36 | 432.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 435.95 | 431.48 | 432.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 436.70 | 431.48 | 432.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 432.70 | 432.61 | 433.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 430.80 | 432.51 | 433.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 435.50 | 432.51 | 432.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 435.50 | 432.51 | 432.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 13:15:00 | 439.80 | 435.54 | 434.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 422.65 | 433.93 | 434.59 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 441.70 | 435.76 | 435.25 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 430.85 | 439.83 | 440.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 422.65 | 431.56 | 436.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 411.15 | 409.45 | 414.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 411.15 | 409.45 | 414.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 418.15 | 411.19 | 415.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 417.40 | 411.19 | 415.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 416.95 | 412.34 | 415.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 414.40 | 412.44 | 415.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 393.68 | 397.01 | 399.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 394.95 | 391.97 | 394.79 | SL hit (close>ema200) qty=0.50 sl=391.97 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 397.25 | 396.17 | 396.04 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 394.55 | 395.99 | 396.01 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 400.40 | 396.87 | 396.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 10:15:00 | 402.95 | 398.80 | 397.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 396.65 | 398.50 | 397.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 397.50 | 398.30 | 397.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 397.10 | 398.30 | 397.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 402.50 | 398.80 | 397.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 403.70 | 399.23 | 398.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 404.00 | 400.11 | 399.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 421.30 | 425.89 | 426.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 421.30 | 425.89 | 426.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 417.90 | 423.32 | 424.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 416.85 | 414.83 | 417.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 416.85 | 414.83 | 417.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 417.55 | 415.37 | 417.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 417.10 | 415.37 | 417.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 416.80 | 415.66 | 417.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 415.30 | 415.75 | 417.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 423.90 | 417.55 | 417.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 423.90 | 417.55 | 417.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 425.20 | 419.08 | 418.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 426.70 | 427.78 | 424.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 426.70 | 427.78 | 424.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 428.05 | 428.35 | 425.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 425.30 | 428.35 | 425.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 427.00 | 427.55 | 425.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 425.50 | 427.55 | 425.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 426.55 | 427.86 | 426.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 433.50 | 427.47 | 426.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 432.30 | 428.28 | 426.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 432.45 | 428.28 | 426.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 424.00 | 426.85 | 426.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 424.00 | 426.85 | 426.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 422.50 | 425.41 | 426.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 426.00 | 424.65 | 425.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 426.00 | 424.92 | 425.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 425.25 | 424.90 | 425.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 425.20 | 425.37 | 425.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 424.10 | 425.37 | 425.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 425.15 | 425.21 | 425.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 424.15 | 425.00 | 425.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 422.90 | 424.48 | 425.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 422.80 | 418.02 | 418.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 423.75 | 419.05 | 418.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 423.75 | 419.05 | 418.83 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 416.10 | 421.53 | 421.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 415.00 | 420.23 | 421.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 412.70 | 407.17 | 409.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 410.60 | 407.86 | 409.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 411.50 | 407.86 | 409.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 408.45 | 407.98 | 409.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 402.55 | 408.05 | 409.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 406.05 | 407.65 | 408.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 407.05 | 407.23 | 408.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 412.60 | 405.02 | 405.75 | SL hit (close>static) qty=1.00 sl=410.60 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 410.20 | 406.96 | 406.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 411.30 | 408.60 | 407.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 409.40 | 409.40 | 408.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 11:15:00 | 414.50 | 410.44 | 408.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 12:45:00 | 414.90 | 411.38 | 409.65 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 14:15:00 | 414.60 | 411.80 | 410.00 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 422.10 | 423.39 | 420.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 419.50 | 423.39 | 420.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 422.50 | 423.39 | 420.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 418.95 | 422.50 | 420.65 | SL hit (close<ema400) qty=1.00 sl=420.65 alert=retest1 |

### Cycle 25 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 428.80 | 432.05 | 432.16 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 435.05 | 432.01 | 432.00 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 15:15:00 | 428.75 | 432.00 | 432.19 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 434.95 | 432.59 | 432.44 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 427.10 | 432.67 | 433.06 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 434.65 | 431.12 | 431.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 438.35 | 433.03 | 431.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 433.30 | 434.39 | 433.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 434.95 | 434.50 | 433.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 436.35 | 434.50 | 433.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 435.25 | 434.65 | 433.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 432.25 | 434.17 | 433.35 | SL hit (close<static) qty=1.00 sl=432.70 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 428.35 | 432.09 | 432.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 418.65 | 428.43 | 430.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 426.80 | 424.75 | 427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 425.40 | 425.08 | 427.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 420.00 | 424.35 | 426.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 419.95 | 424.35 | 426.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 419.25 | 417.66 | 420.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 420.25 | 418.18 | 420.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 419.45 | 418.43 | 420.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 420.90 | 418.43 | 420.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 418.10 | 418.54 | 420.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 422.75 | 420.82 | 420.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 422.75 | 420.82 | 420.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 437.00 | 424.72 | 422.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 467.15 | 467.30 | 455.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:30:00 | 465.80 | 467.30 | 455.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 449.10 | 463.38 | 461.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 454.75 | 463.38 | 461.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 447.80 | 460.26 | 460.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 448.00 | 460.26 | 460.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 446.00 | 457.41 | 458.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 13:15:00 | 438.75 | 451.50 | 455.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 449.90 | 445.85 | 451.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 449.90 | 445.85 | 451.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 447.00 | 446.08 | 450.73 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 468.30 | 456.24 | 454.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 474.75 | 459.94 | 456.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 478.25 | 479.25 | 471.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 478.25 | 479.25 | 471.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 480.30 | 481.33 | 478.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 480.30 | 481.33 | 478.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 484.10 | 485.05 | 482.62 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 467.00 | 479.54 | 480.70 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 496.35 | 481.60 | 479.80 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 485.60 | 486.17 | 486.17 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 491.60 | 487.25 | 486.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 504.20 | 491.36 | 489.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 498.00 | 499.60 | 495.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:00:00 | 498.00 | 499.60 | 495.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 493.25 | 498.33 | 495.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 493.45 | 498.33 | 495.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 491.45 | 496.96 | 495.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 491.45 | 496.96 | 495.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 496.00 | 496.76 | 495.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 496.55 | 495.17 | 494.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 490.80 | 494.30 | 494.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 490.80 | 494.30 | 494.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 488.30 | 493.10 | 493.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 497.20 | 491.05 | 492.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 501.30 | 493.10 | 493.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 501.30 | 493.10 | 493.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 498.80 | 494.24 | 493.70 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 487.80 | 493.30 | 493.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 482.00 | 489.45 | 491.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 461.25 | 460.53 | 467.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:30:00 | 461.20 | 460.53 | 467.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 477.85 | 459.86 | 461.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 477.85 | 459.86 | 461.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 485.00 | 464.89 | 463.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 491.25 | 477.95 | 471.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 490.95 | 494.03 | 485.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 484.45 | 492.11 | 485.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 484.45 | 492.11 | 485.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 484.45 | 492.11 | 485.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 482.95 | 490.28 | 485.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 481.75 | 490.28 | 485.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 480.10 | 488.24 | 484.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 480.10 | 488.24 | 484.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 475.70 | 485.73 | 483.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 475.70 | 485.73 | 483.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 475.10 | 482.23 | 482.59 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 489.35 | 483.35 | 483.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 493.25 | 487.32 | 485.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 486.15 | 489.28 | 489.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 483.30 | 487.11 | 487.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 482.15 | 483.19 | 484.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 486.00 | 484.30 | 484.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 486.00 | 484.30 | 484.16 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 477.50 | 482.88 | 483.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 475.50 | 481.40 | 482.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 12:15:00 | 471.95 | 470.66 | 474.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 471.95 | 470.66 | 474.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 459.80 | 462.97 | 467.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 458.50 | 462.03 | 466.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 458.25 | 462.03 | 466.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 458.10 | 461.53 | 465.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 457.25 | 461.02 | 464.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 458.70 | 459.15 | 461.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 458.70 | 459.15 | 461.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 460.50 | 459.42 | 461.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 457.45 | 459.01 | 460.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 461.80 | 459.36 | 459.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 461.80 | 459.36 | 459.35 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 457.90 | 459.60 | 459.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 456.70 | 459.02 | 459.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 462.90 | 456.78 | 457.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 459.80 | 457.38 | 457.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 463.70 | 457.38 | 457.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 466.00 | 459.11 | 458.58 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 452.15 | 458.82 | 459.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 447.90 | 456.64 | 458.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 441.00 | 440.13 | 445.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 442.60 | 440.13 | 445.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 440.60 | 440.23 | 444.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 438.05 | 439.78 | 443.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 437.75 | 438.75 | 442.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 437.65 | 438.81 | 441.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 437.20 | 438.59 | 441.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 438.40 | 438.55 | 441.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 438.40 | 438.55 | 441.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 434.80 | 434.30 | 437.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 436.95 | 434.30 | 437.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 434.55 | 433.38 | 435.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 434.55 | 433.38 | 435.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 416.15 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 415.86 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 415.77 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 415.34 | 420.77 | 426.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 394.25 | 406.48 | 416.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 402.30 | 398.97 | 398.81 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 394.05 | 397.85 | 398.36 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 403.25 | 399.20 | 398.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 12:15:00 | 405.25 | 401.11 | 399.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 453.75 | 461.58 | 448.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 446.10 | 458.49 | 448.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 446.10 | 458.49 | 448.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 440.10 | 454.81 | 447.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 440.10 | 454.81 | 447.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 435.80 | 443.00 | 443.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 432.45 | 439.54 | 441.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 394.55 | 393.52 | 400.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 394.55 | 393.52 | 400.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 397.30 | 390.07 | 393.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 397.30 | 390.07 | 393.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 396.15 | 391.28 | 393.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 398.05 | 391.28 | 393.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 399.60 | 392.95 | 394.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 401.75 | 392.95 | 394.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 393.75 | 394.06 | 394.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 393.75 | 394.06 | 394.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 395.75 | 394.40 | 394.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 396.00 | 394.40 | 394.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 390.70 | 393.66 | 394.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 398.55 | 393.66 | 394.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 392.40 | 392.60 | 393.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 393.65 | 392.60 | 393.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 391.50 | 392.28 | 393.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 387.90 | 391.30 | 392.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 386.75 | 390.71 | 392.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 387.70 | 390.71 | 392.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 368.50 | 379.84 | 385.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 368.31 | 379.84 | 385.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 367.41 | 377.30 | 383.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-23 13:15:00 | 349.11 | 358.36 | 369.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 286.35 | 283.58 | 283.27 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 278.45 | 282.70 | 282.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 274.70 | 281.10 | 282.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 256.15 | 255.87 | 262.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 256.15 | 255.87 | 262.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 263.85 | 258.02 | 262.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 263.35 | 258.02 | 262.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 263.10 | 259.04 | 262.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 264.30 | 259.04 | 262.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 263.00 | 260.52 | 262.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 270.25 | 260.52 | 262.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 272.00 | 262.82 | 263.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 272.20 | 262.82 | 263.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 272.85 | 264.82 | 264.11 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 263.15 | 266.14 | 266.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 260.65 | 264.43 | 265.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 269.05 | 257.99 | 260.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 261.05 | 258.60 | 260.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 259.65 | 260.37 | 260.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 260.00 | 260.37 | 260.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 264.10 | 261.23 | 261.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 264.10 | 261.23 | 261.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 09:15:00 | 278.30 | 264.64 | 262.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 15:15:00 | 298.80 | 299.25 | 290.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 09:15:00 | 295.00 | 299.25 | 290.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 294.45 | 298.29 | 290.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 297.00 | 295.17 | 292.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 297.00 | 298.62 | 295.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 298.15 | 298.17 | 296.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 297.10 | 297.37 | 296.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 296.20 | 297.13 | 296.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 296.35 | 297.13 | 296.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 296.40 | 296.99 | 296.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 13:30:00 | 296.30 | 296.99 | 296.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 295.80 | 296.75 | 296.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 295.45 | 296.75 | 296.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 295.15 | 296.43 | 296.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 295.15 | 296.43 | 296.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 288.65 | 294.87 | 295.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 300.50 | 292.98 | 293.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 300.85 | 294.56 | 294.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 304.15 | 296.48 | 295.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 310.25 | 310.64 | 307.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 305.35 | 310.64 | 307.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 305.65 | 309.64 | 307.24 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 302.00 | 306.03 | 306.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 296.30 | 303.01 | 304.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 294.10 | 291.27 | 294.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 296.05 | 292.22 | 294.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 295.50 | 292.22 | 294.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 297.20 | 293.22 | 294.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 296.70 | 293.22 | 294.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 297.50 | 294.07 | 295.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 296.00 | 295.50 | 295.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 297.20 | 295.84 | 295.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 297.20 | 295.84 | 295.72 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 295.15 | 295.66 | 295.66 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 295.85 | 295.70 | 295.68 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 294.90 | 295.54 | 295.61 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 296.75 | 295.78 | 295.71 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 294.45 | 295.80 | 295.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 290.45 | 294.73 | 295.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 293.20 | 293.27 | 294.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 292.40 | 293.10 | 293.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 291.55 | 292.70 | 293.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 293.60 | 292.35 | 292.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 293.60 | 292.35 | 292.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 296.80 | 293.24 | 292.73 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-29 09:15:00 | 422.45 | 2025-05-29 10:15:00 | 417.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-05-29 14:30:00 | 422.35 | 2025-05-30 09:15:00 | 415.20 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-06-03 13:15:00 | 413.35 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-06-03 14:00:00 | 414.30 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-06-04 09:15:00 | 414.45 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-06-25 09:15:00 | 408.85 | 2025-07-03 14:15:00 | 449.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 11:00:00 | 408.50 | 2025-07-03 14:15:00 | 449.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-16 10:15:00 | 430.80 | 2025-07-17 10:15:00 | 435.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-30 09:30:00 | 414.40 | 2025-08-08 15:15:00 | 393.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:30:00 | 414.40 | 2025-08-12 09:15:00 | 394.95 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2025-08-18 10:15:00 | 403.70 | 2025-08-28 14:15:00 | 421.30 | STOP_HIT | 1.00 | 4.36% |
| BUY | retest2 | 2025-08-19 10:15:00 | 404.00 | 2025-08-28 14:15:00 | 421.30 | STOP_HIT | 1.00 | 4.28% |
| SELL | retest2 | 2025-09-02 13:15:00 | 415.30 | 2025-09-03 12:15:00 | 423.90 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-08 11:15:00 | 433.50 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-09-08 11:45:00 | 432.30 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-09-08 12:15:00 | 432.45 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-10 11:30:00 | 425.25 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-09-10 14:45:00 | 425.20 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-09-10 15:15:00 | 424.10 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-09-11 10:15:00 | 425.15 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-09-11 11:30:00 | 422.90 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-09-17 09:30:00 | 422.80 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-26 09:15:00 | 402.55 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-09-26 10:00:00 | 406.05 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-26 13:15:00 | 407.05 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2025-10-03 11:15:00 | 414.50 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2025-10-03 12:45:00 | 414.90 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest1 | 2025-10-03 14:15:00 | 414.60 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-10-09 09:15:00 | 426.70 | 2025-10-14 13:15:00 | 428.80 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-10-24 09:15:00 | 436.35 | 2025-10-24 10:15:00 | 432.25 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-24 10:00:00 | 435.25 | 2025-10-24 10:15:00 | 432.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 10:30:00 | 420.00 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-28 11:15:00 | 419.95 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-29 13:15:00 | 419.25 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-29 14:00:00 | 420.25 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-27 11:15:00 | 496.55 | 2025-11-27 11:15:00 | 490.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-19 11:30:00 | 482.15 | 2025-12-22 13:15:00 | 486.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-29 10:30:00 | 458.50 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-29 11:00:00 | 458.25 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 12:15:00 | 458.10 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-30 09:15:00 | 457.25 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-01 12:00:00 | 457.45 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-13 12:00:00 | 438.05 | 2026-01-20 10:15:00 | 416.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 437.75 | 2026-01-20 10:15:00 | 415.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 437.65 | 2026-01-20 10:15:00 | 415.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 437.20 | 2026-01-20 11:15:00 | 415.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 438.05 | 2026-01-21 10:15:00 | 394.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 437.75 | 2026-01-21 10:15:00 | 393.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 437.65 | 2026-01-21 10:15:00 | 393.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 437.20 | 2026-01-21 10:15:00 | 393.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 399.75 | 2026-01-23 13:15:00 | 379.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 399.75 | 2026-01-27 12:15:00 | 395.85 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2026-01-28 12:15:00 | 397.15 | 2026-01-28 14:15:00 | 402.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-02-19 11:30:00 | 387.90 | 2026-02-20 11:15:00 | 368.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 386.75 | 2026-02-20 11:15:00 | 368.31 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-02-19 13:15:00 | 387.70 | 2026-02-20 12:15:00 | 367.41 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-02-19 11:30:00 | 387.90 | 2026-02-23 13:15:00 | 349.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 386.75 | 2026-02-23 13:15:00 | 348.93 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-02-19 13:15:00 | 387.70 | 2026-02-24 09:15:00 | 348.07 | TARGET_HIT | 0.50 | 10.22% |
| SELL | retest2 | 2026-04-01 13:30:00 | 259.65 | 2026-04-01 15:15:00 | 264.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-01 14:15:00 | 260.00 | 2026-04-01 15:15:00 | 264.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-04-08 09:30:00 | 297.00 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-09 10:15:00 | 297.00 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-09 14:45:00 | 298.15 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-10 11:30:00 | 297.10 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-28 09:45:00 | 296.00 | 2026-04-28 10:15:00 | 297.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-05-04 10:45:00 | 293.20 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-05-04 12:00:00 | 292.40 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-05-05 09:15:00 | 291.55 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.70% |
