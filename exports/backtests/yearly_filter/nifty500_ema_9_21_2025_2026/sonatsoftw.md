# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 296.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 30 |
| ALERT3 | 140 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 62 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 40
- **Target hits / Stop hits / Partials:** 7 / 57 / 7
- **Avg / median % per leg:** 0.85% / -0.75%
- **Sum % (uncompounded):** 60.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 16 | 45.7% | 5 | 30 | 0 | 0.78% | 27.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 16 | 45.7% | 5 | 30 | 0 | 0.78% | 27.2% |
| SELL (all) | 36 | 15 | 41.7% | 2 | 27 | 7 | 0.92% | 33.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.18% | 6.5% |
| SELL @ 3rd Alert (retest2) | 33 | 13 | 39.4% | 2 | 25 | 6 | 0.80% | 26.5% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.18% | 6.5% |
| retest2 (combined) | 68 | 29 | 42.6% | 7 | 55 | 6 | 0.79% | 53.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 394.15 | 390.44 | 389.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 399.10 | 392.17 | 390.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 396.00 | 398.06 | 395.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 396.00 | 398.06 | 395.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 396.00 | 398.06 | 395.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 396.00 | 398.06 | 395.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 395.95 | 397.64 | 395.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 395.95 | 397.64 | 395.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 395.85 | 397.28 | 395.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 395.50 | 397.28 | 395.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 395.50 | 396.93 | 395.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 393.50 | 396.93 | 395.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 394.05 | 396.35 | 395.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 394.05 | 396.35 | 395.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 392.75 | 395.29 | 395.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 397.50 | 395.29 | 395.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 11:15:00 | 393.80 | 396.75 | 396.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 393.80 | 396.75 | 396.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 12:15:00 | 390.75 | 395.55 | 396.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 394.50 | 392.95 | 394.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 394.50 | 392.95 | 394.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 394.50 | 392.95 | 394.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 394.50 | 392.95 | 394.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 393.50 | 393.06 | 394.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 11:30:00 | 392.45 | 393.04 | 394.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 12:15:00 | 391.95 | 393.04 | 394.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:15:00 | 392.60 | 393.50 | 394.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 392.55 | 393.02 | 393.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 392.35 | 392.88 | 393.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 390.30 | 392.37 | 393.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 394.20 | 392.68 | 393.30 | SL hit (close>static) qty=1.00 sl=394.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 400.40 | 392.83 | 392.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 405.25 | 397.06 | 394.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 398.90 | 399.10 | 396.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 396.75 | 399.10 | 396.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 398.20 | 398.92 | 397.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 401.00 | 399.82 | 398.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 409.00 | 412.30 | 412.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 409.00 | 412.30 | 412.53 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 418.90 | 412.13 | 411.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 422.85 | 415.37 | 413.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 414.40 | 415.90 | 414.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 15:00:00 | 414.40 | 415.90 | 414.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 416.50 | 416.02 | 414.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 426.30 | 416.02 | 414.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 417.60 | 421.66 | 420.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 414.40 | 419.14 | 419.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 414.40 | 419.14 | 419.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 412.45 | 416.08 | 417.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 414.10 | 413.23 | 415.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 414.10 | 413.23 | 415.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 416.00 | 414.00 | 415.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 415.40 | 414.00 | 415.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 424.25 | 416.05 | 416.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 424.25 | 416.05 | 416.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 425.95 | 418.03 | 417.18 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 415.75 | 419.11 | 419.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 410.65 | 416.79 | 418.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 12:15:00 | 395.50 | 394.65 | 398.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 395.50 | 394.65 | 398.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 408.00 | 397.10 | 398.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 408.00 | 397.10 | 398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 414.05 | 400.49 | 399.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 424.20 | 413.05 | 409.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 414.90 | 415.20 | 411.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:15:00 | 414.50 | 415.20 | 411.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 412.50 | 414.66 | 411.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 412.50 | 414.66 | 411.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 413.00 | 414.33 | 411.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 414.80 | 414.33 | 411.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 410.25 | 412.46 | 411.68 | SL hit (close<static) qty=1.00 sl=411.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 405.80 | 410.39 | 410.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 404.80 | 408.14 | 409.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 407.45 | 406.51 | 407.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 407.45 | 406.51 | 407.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 407.45 | 406.51 | 407.77 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 409.60 | 407.63 | 407.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 412.05 | 409.70 | 408.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 432.30 | 434.15 | 427.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 432.30 | 434.15 | 427.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 429.30 | 432.86 | 429.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 429.30 | 432.86 | 429.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 426.20 | 431.53 | 429.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 426.20 | 431.53 | 429.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 428.00 | 430.82 | 429.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 428.70 | 430.82 | 429.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 428.85 | 430.43 | 429.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:00:00 | 429.20 | 428.99 | 428.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 429.25 | 429.45 | 429.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 428.95 | 429.35 | 429.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 433.05 | 430.43 | 429.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 436.80 | 431.35 | 430.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 434.70 | 436.17 | 435.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 12:15:00 | 434.85 | 435.33 | 435.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 434.85 | 435.33 | 435.38 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 437.35 | 435.74 | 435.56 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 429.40 | 434.89 | 435.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 423.75 | 430.42 | 432.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 428.60 | 427.74 | 430.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 428.60 | 427.74 | 430.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 430.10 | 428.25 | 430.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 424.00 | 428.13 | 429.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 402.80 | 409.01 | 414.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 408.50 | 405.83 | 410.55 | SL hit (close>ema200) qty=0.50 sl=405.83 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 414.20 | 411.99 | 411.78 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 397.55 | 409.10 | 410.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 387.35 | 404.75 | 408.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 350.00 | 349.76 | 354.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:30:00 | 349.90 | 349.76 | 354.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 360.90 | 339.85 | 342.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 360.90 | 339.85 | 342.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 366.20 | 345.12 | 344.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 375.90 | 359.38 | 352.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 355.75 | 360.84 | 355.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 355.75 | 360.84 | 355.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 355.75 | 360.84 | 355.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 355.75 | 360.84 | 355.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 362.85 | 363.65 | 360.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 361.00 | 363.65 | 360.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 359.50 | 362.82 | 360.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 359.50 | 362.82 | 360.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 358.40 | 361.94 | 360.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 355.00 | 361.94 | 360.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 352.80 | 358.77 | 358.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 348.50 | 356.71 | 358.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 356.70 | 356.41 | 357.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 356.70 | 356.41 | 357.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 356.70 | 356.41 | 357.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 356.70 | 356.41 | 357.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 356.10 | 356.21 | 357.02 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 364.00 | 357.67 | 357.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 366.90 | 360.69 | 358.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 372.70 | 374.90 | 370.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 372.70 | 374.90 | 370.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 372.70 | 374.90 | 370.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 372.45 | 374.90 | 370.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 370.20 | 373.96 | 370.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 369.90 | 373.96 | 370.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 373.05 | 373.78 | 370.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 374.40 | 373.75 | 371.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 387.95 | 373.58 | 371.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 367.70 | 376.33 | 376.20 | SL hit (close<static) qty=1.00 sl=369.75 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 368.00 | 374.66 | 375.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 364.80 | 372.69 | 374.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 363.00 | 361.33 | 365.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 363.00 | 361.33 | 365.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 360.55 | 357.03 | 361.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 362.60 | 357.03 | 361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 359.10 | 357.45 | 360.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 355.00 | 357.45 | 360.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 356.90 | 356.30 | 357.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 361.00 | 357.91 | 357.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 361.00 | 357.91 | 357.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 364.55 | 359.24 | 358.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 360.80 | 362.08 | 360.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:00:00 | 360.80 | 362.08 | 360.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 360.00 | 361.67 | 360.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 360.00 | 361.67 | 360.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 359.60 | 361.25 | 360.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 358.85 | 361.25 | 360.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 359.05 | 360.81 | 360.05 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 359.10 | 359.65 | 359.69 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 360.75 | 359.87 | 359.78 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 358.05 | 359.51 | 359.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 354.85 | 358.58 | 359.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 358.50 | 358.29 | 358.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 358.50 | 358.29 | 358.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 358.50 | 358.29 | 358.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 360.65 | 358.29 | 358.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 360.65 | 358.76 | 359.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 360.65 | 358.76 | 359.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 365.20 | 360.05 | 359.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 377.50 | 367.32 | 363.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 15:15:00 | 371.00 | 371.28 | 367.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:15:00 | 370.55 | 371.28 | 367.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 378.60 | 380.32 | 378.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:45:00 | 378.50 | 380.32 | 378.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 380.25 | 380.31 | 378.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:45:00 | 378.60 | 380.31 | 378.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 377.25 | 379.88 | 378.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 377.25 | 379.88 | 378.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 377.40 | 379.38 | 378.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 376.40 | 379.38 | 378.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 379.70 | 379.26 | 378.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 385.90 | 379.26 | 378.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 389.25 | 392.89 | 392.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 389.25 | 392.89 | 392.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 380.40 | 387.17 | 389.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 370.05 | 369.82 | 375.55 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:15:00 | 361.80 | 369.82 | 375.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 343.71 | 351.42 | 359.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 348.85 | 350.10 | 357.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 350.20 | 349.93 | 355.28 | SL hit (close>ema200) qty=0.50 sl=349.93 alert=retest1 |

### Cycle 27 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 351.50 | 350.30 | 350.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 356.00 | 351.44 | 350.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 14:15:00 | 351.55 | 352.71 | 351.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 351.55 | 352.71 | 351.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 351.55 | 352.71 | 351.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 351.55 | 352.71 | 351.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 351.90 | 352.55 | 351.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 355.65 | 352.55 | 351.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 353.15 | 352.39 | 351.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 353.00 | 352.58 | 352.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 361.35 | 367.76 | 367.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 361.35 | 367.76 | 367.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 359.70 | 363.57 | 365.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 364.00 | 362.11 | 363.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 364.00 | 362.11 | 363.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 364.00 | 362.11 | 363.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 364.50 | 362.11 | 363.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 361.05 | 361.90 | 363.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 368.60 | 361.90 | 363.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 364.95 | 362.51 | 363.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:00:00 | 362.70 | 362.83 | 363.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:00:00 | 362.60 | 362.79 | 363.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 371.20 | 364.39 | 364.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 371.20 | 364.39 | 364.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 361.30 | 365.85 | 366.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 360.50 | 363.52 | 364.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 365.80 | 362.44 | 363.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 365.80 | 362.44 | 363.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 365.80 | 362.44 | 363.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 365.80 | 362.44 | 363.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 364.80 | 362.91 | 363.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 374.05 | 362.91 | 363.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 378.00 | 365.93 | 365.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 387.05 | 373.13 | 368.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 373.85 | 376.48 | 371.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 373.85 | 376.48 | 371.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 371.80 | 375.55 | 371.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 378.15 | 375.55 | 371.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 374.10 | 375.00 | 373.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 374.15 | 374.85 | 373.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 374.05 | 374.68 | 373.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 374.10 | 374.56 | 373.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 374.00 | 374.56 | 373.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 373.85 | 374.42 | 373.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 372.55 | 374.42 | 373.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 370.95 | 373.73 | 373.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 370.95 | 373.73 | 373.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 368.55 | 372.69 | 373.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 368.55 | 372.69 | 373.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 367.00 | 371.55 | 372.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 370.65 | 370.64 | 371.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 370.65 | 370.64 | 371.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 371.10 | 370.40 | 371.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 370.90 | 370.40 | 371.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 373.20 | 370.96 | 371.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 372.40 | 370.96 | 371.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 372.60 | 371.29 | 371.69 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 373.65 | 372.04 | 371.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 374.80 | 372.59 | 372.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 371.50 | 372.37 | 372.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 371.50 | 372.37 | 372.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 371.50 | 372.37 | 372.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 371.50 | 372.37 | 372.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 371.25 | 372.15 | 372.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 370.50 | 372.15 | 372.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 371.50 | 372.02 | 372.03 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 372.90 | 372.17 | 372.09 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 369.60 | 371.64 | 371.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 368.80 | 370.82 | 371.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 371.10 | 370.29 | 370.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 15:15:00 | 371.10 | 370.29 | 370.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 371.10 | 370.29 | 370.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 372.60 | 370.29 | 370.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 372.75 | 370.78 | 371.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 372.75 | 370.78 | 371.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 372.60 | 371.15 | 371.28 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 373.05 | 371.53 | 371.44 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 369.55 | 371.33 | 371.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 369.40 | 370.48 | 370.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 14:15:00 | 375.30 | 371.16 | 371.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 375.30 | 371.16 | 371.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 375.30 | 371.16 | 371.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 375.30 | 371.16 | 371.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 372.00 | 371.33 | 371.26 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 365.35 | 370.13 | 370.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 359.55 | 365.39 | 367.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 366.20 | 364.76 | 366.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 366.20 | 364.76 | 366.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 366.20 | 364.76 | 366.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 368.20 | 364.76 | 366.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 367.15 | 365.24 | 366.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 367.15 | 365.24 | 366.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 367.75 | 365.74 | 366.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:30:00 | 365.50 | 365.71 | 366.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 374.50 | 367.47 | 367.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 374.50 | 367.47 | 367.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 383.35 | 375.00 | 372.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 391.20 | 392.31 | 386.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 391.20 | 392.31 | 386.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 389.00 | 391.65 | 386.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 369.30 | 391.65 | 386.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 369.50 | 387.22 | 385.39 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 375.40 | 382.84 | 383.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 371.85 | 379.06 | 381.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 369.60 | 367.32 | 370.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 369.60 | 367.32 | 370.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 369.60 | 367.32 | 370.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 369.60 | 367.32 | 370.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 369.35 | 367.73 | 370.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 369.85 | 367.73 | 370.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 369.80 | 368.14 | 370.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 370.30 | 368.14 | 370.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 368.55 | 368.22 | 369.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:15:00 | 371.50 | 368.22 | 369.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 370.35 | 368.65 | 369.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 367.40 | 369.08 | 369.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 361.60 | 360.10 | 360.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 361.60 | 360.10 | 360.08 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 358.70 | 359.88 | 359.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 357.95 | 359.26 | 359.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 364.80 | 359.26 | 359.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 364.80 | 359.26 | 359.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 364.80 | 359.26 | 359.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 364.80 | 359.26 | 359.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 360.80 | 359.57 | 359.52 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 357.60 | 359.17 | 359.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 352.80 | 356.71 | 357.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 358.05 | 354.34 | 356.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 358.05 | 354.34 | 356.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 358.05 | 354.34 | 356.09 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 368.85 | 359.49 | 358.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 372.60 | 364.84 | 361.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 14:15:00 | 365.60 | 366.67 | 363.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:45:00 | 366.85 | 366.67 | 363.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 358.70 | 364.92 | 363.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 358.70 | 364.92 | 363.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 363.00 | 364.54 | 363.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 362.85 | 364.54 | 363.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 364.20 | 364.47 | 363.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 364.00 | 364.47 | 363.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 363.45 | 364.27 | 363.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 363.25 | 364.27 | 363.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 363.00 | 364.01 | 363.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 363.05 | 364.01 | 363.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 360.20 | 363.25 | 363.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 357.25 | 361.53 | 362.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 352.65 | 351.44 | 354.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 352.65 | 351.44 | 354.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 352.65 | 351.44 | 354.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 354.75 | 351.44 | 354.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 347.65 | 347.13 | 349.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 348.85 | 347.13 | 349.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 351.60 | 348.31 | 350.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 351.60 | 348.31 | 350.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 351.40 | 348.93 | 350.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 351.60 | 348.93 | 350.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 351.30 | 350.90 | 350.86 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 13:15:00 | 350.00 | 350.72 | 350.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 348.90 | 350.17 | 350.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 343.05 | 340.79 | 342.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 343.05 | 340.79 | 342.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 343.05 | 340.79 | 342.36 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 345.70 | 343.21 | 342.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 355.65 | 345.69 | 344.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 357.75 | 358.20 | 354.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 357.55 | 358.20 | 354.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 355.70 | 357.58 | 354.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 355.70 | 357.58 | 354.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 358.05 | 357.68 | 355.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 355.25 | 357.68 | 355.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 366.35 | 359.81 | 356.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:30:00 | 369.50 | 362.67 | 358.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 353.90 | 359.92 | 360.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 353.90 | 359.92 | 360.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 353.25 | 357.85 | 358.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 358.30 | 357.60 | 358.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 12:00:00 | 358.30 | 357.60 | 358.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 360.20 | 358.12 | 358.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 360.20 | 358.12 | 358.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 361.45 | 358.79 | 359.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 361.45 | 358.79 | 359.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 357.55 | 358.10 | 358.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 361.75 | 358.10 | 358.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 360.45 | 358.57 | 358.82 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 359.50 | 358.99 | 358.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 360.90 | 359.63 | 359.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 361.00 | 361.41 | 360.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 361.00 | 361.41 | 360.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 361.00 | 361.41 | 360.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:30:00 | 364.25 | 362.54 | 361.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 364.90 | 362.54 | 361.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 363.90 | 363.58 | 362.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 354.55 | 360.59 | 361.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 354.55 | 360.59 | 361.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 353.20 | 359.11 | 360.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 362.55 | 355.66 | 357.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 362.55 | 355.66 | 357.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 362.55 | 355.66 | 357.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 362.55 | 355.66 | 357.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 366.40 | 357.81 | 358.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 367.90 | 357.81 | 358.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 368.00 | 359.85 | 359.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 373.80 | 365.75 | 362.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 364.00 | 366.40 | 363.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 364.00 | 366.40 | 363.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 364.00 | 366.40 | 363.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 364.00 | 366.40 | 363.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 360.85 | 365.29 | 363.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 360.85 | 365.29 | 363.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 360.30 | 364.29 | 362.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 360.30 | 364.29 | 362.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 361.20 | 362.38 | 362.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 355.40 | 362.38 | 362.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 356.10 | 361.12 | 361.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 353.45 | 358.59 | 360.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 349.00 | 348.65 | 352.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 348.25 | 348.65 | 352.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 348.40 | 348.60 | 352.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 345.80 | 349.32 | 350.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 345.60 | 349.32 | 350.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 355.35 | 349.73 | 350.29 | SL hit (close>static) qty=1.00 sl=352.80 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 311.80 | 311.19 | 311.17 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 309.55 | 311.08 | 311.15 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 316.00 | 311.96 | 311.47 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 309.55 | 313.18 | 313.65 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 324.95 | 315.39 | 314.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 328.45 | 319.65 | 316.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 318.20 | 322.40 | 319.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 318.20 | 322.40 | 319.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 318.20 | 322.40 | 319.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 317.25 | 322.40 | 319.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 312.55 | 320.43 | 318.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 312.55 | 320.43 | 318.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 314.90 | 317.36 | 317.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 311.85 | 315.73 | 316.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 307.00 | 306.18 | 309.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:15:00 | 310.25 | 306.18 | 309.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 306.25 | 306.19 | 308.81 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 313.95 | 310.31 | 310.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 314.95 | 311.24 | 310.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 315.65 | 317.20 | 315.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 315.65 | 317.20 | 315.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 315.65 | 317.20 | 315.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 315.00 | 317.20 | 315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 314.85 | 316.47 | 315.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 314.80 | 316.47 | 315.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 315.40 | 316.26 | 315.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 315.55 | 316.26 | 315.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 316.35 | 316.28 | 315.20 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 307.05 | 314.02 | 314.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 300.15 | 309.82 | 312.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 283.85 | 277.40 | 283.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 283.85 | 277.40 | 283.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 283.85 | 277.40 | 283.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 283.85 | 277.40 | 283.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 282.50 | 278.42 | 283.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 282.85 | 278.42 | 283.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 287.45 | 280.22 | 283.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 284.80 | 280.22 | 283.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 287.30 | 281.64 | 284.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 288.65 | 281.64 | 284.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 284.05 | 282.47 | 284.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:45:00 | 284.30 | 282.47 | 284.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 284.00 | 282.77 | 284.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 282.85 | 282.77 | 284.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 279.25 | 282.07 | 283.75 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 288.00 | 284.31 | 283.84 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 282.60 | 285.05 | 285.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 280.75 | 284.19 | 284.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 276.10 | 274.40 | 278.08 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 12:30:00 | 272.80 | 274.13 | 277.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 277.35 | 274.02 | 275.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 277.35 | 274.02 | 275.98 | SL hit (close>ema400) qty=1.00 sl=275.98 alert=retest1 |

### Cycle 67 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 240.40 | 236.89 | 236.56 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 233.55 | 236.68 | 237.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 231.30 | 235.05 | 236.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 238.00 | 234.72 | 235.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 238.00 | 234.72 | 235.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 238.00 | 234.72 | 235.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 238.00 | 234.72 | 235.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 236.45 | 235.06 | 235.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:15:00 | 239.85 | 235.06 | 235.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 237.55 | 236.16 | 236.14 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 231.05 | 235.37 | 235.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 229.20 | 234.14 | 235.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 228.90 | 227.45 | 230.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 228.90 | 227.45 | 230.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 229.85 | 227.93 | 230.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 229.85 | 227.93 | 230.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 229.75 | 228.29 | 230.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 234.20 | 228.29 | 230.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 227.90 | 228.22 | 229.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:30:00 | 225.30 | 227.57 | 228.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 214.03 | 220.50 | 224.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 223.01 | 215.57 | 219.16 | SL hit (close>ema200) qty=0.50 sl=215.57 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 223.92 | 220.87 | 220.85 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 15:15:00 | 220.00 | 220.69 | 220.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 214.14 | 219.38 | 220.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 11:15:00 | 220.00 | 219.10 | 219.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 11:15:00 | 220.00 | 219.10 | 219.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 220.00 | 219.10 | 219.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:45:00 | 221.19 | 219.10 | 219.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 224.10 | 220.10 | 220.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 224.10 | 220.10 | 220.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 228.96 | 221.87 | 221.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 232.90 | 224.08 | 222.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 244.82 | 245.31 | 241.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 244.82 | 245.31 | 241.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 241.14 | 245.31 | 243.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 244.43 | 244.94 | 243.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 243.71 | 244.94 | 243.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 245.44 | 243.94 | 243.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 248.62 | 243.49 | 243.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 268.87 | 258.83 | 251.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 268.00 | 271.92 | 272.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 267.28 | 270.99 | 271.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 269.60 | 269.32 | 270.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 269.60 | 269.32 | 270.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 269.60 | 269.32 | 270.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 269.60 | 269.32 | 270.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 271.90 | 269.83 | 270.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 271.90 | 269.83 | 270.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 271.01 | 270.07 | 270.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:15:00 | 270.00 | 270.07 | 270.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 269.78 | 269.79 | 270.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 256.50 | 263.07 | 266.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 256.29 | 263.07 | 266.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 259.82 | 259.08 | 261.92 | SL hit (close>ema200) qty=0.50 sl=259.08 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 263.81 | 260.94 | 260.81 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 258.40 | 260.66 | 260.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 257.00 | 259.93 | 260.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 258.45 | 257.67 | 259.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 258.45 | 257.67 | 259.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 258.45 | 257.67 | 259.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 258.45 | 257.67 | 259.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 263.40 | 258.81 | 259.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 263.40 | 258.81 | 259.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 260.45 | 259.14 | 259.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 259.15 | 259.44 | 259.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 261.10 | 259.97 | 259.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 261.10 | 259.97 | 259.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 262.00 | 260.37 | 260.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 259.85 | 260.41 | 260.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 259.85 | 260.41 | 260.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 259.85 | 260.41 | 260.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 259.85 | 260.41 | 260.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 259.50 | 260.23 | 260.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 258.70 | 260.23 | 260.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 259.25 | 260.03 | 259.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 259.25 | 260.03 | 259.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 260.50 | 260.12 | 260.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 261.60 | 260.12 | 260.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 09:15:00 | 287.76 | 274.25 | 269.34 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 13:15:00 | 394.05 | 2025-05-12 14:15:00 | 394.15 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-05-12 14:15:00 | 394.00 | 2025-05-12 14:15:00 | 394.15 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-15 09:15:00 | 397.50 | 2025-05-19 11:15:00 | 393.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-20 11:30:00 | 392.45 | 2025-05-21 14:15:00 | 394.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-20 12:15:00 | 391.95 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-05-20 14:15:00 | 392.60 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-05-21 10:00:00 | 392.55 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-05-21 12:00:00 | 390.30 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-05-22 12:15:00 | 390.70 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-05-23 09:15:00 | 389.55 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-05-27 15:15:00 | 401.00 | 2025-06-05 15:15:00 | 409.00 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-06-10 09:15:00 | 426.30 | 2025-06-12 15:15:00 | 414.40 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-06-12 14:15:00 | 417.60 | 2025-06-12 15:15:00 | 414.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-30 09:15:00 | 414.80 | 2025-06-30 12:15:00 | 410.25 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-10 13:15:00 | 428.70 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-07-10 14:00:00 | 428.85 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-11 14:00:00 | 429.20 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-07-14 10:00:00 | 429.25 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-07-14 14:15:00 | 433.05 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-07-15 09:15:00 | 436.80 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-17 11:00:00 | 434.70 | 2025-07-18 12:15:00 | 434.85 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-07-25 09:30:00 | 424.00 | 2025-07-29 09:15:00 | 402.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 424.00 | 2025-07-29 14:15:00 | 408.50 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2025-08-22 13:30:00 | 374.40 | 2025-08-26 11:15:00 | 367.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-25 09:15:00 | 387.95 | 2025-08-26 11:15:00 | 367.70 | STOP_HIT | 1.00 | -5.22% |
| SELL | retest2 | 2025-09-01 11:15:00 | 355.00 | 2025-09-03 15:15:00 | 361.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-09-03 09:30:00 | 356.90 | 2025-09-03 15:15:00 | 361.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-17 09:15:00 | 385.90 | 2025-09-22 10:15:00 | 389.25 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest1 | 2025-09-26 09:15:00 | 361.80 | 2025-09-29 14:15:00 | 343.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-26 09:15:00 | 361.80 | 2025-09-30 12:15:00 | 350.20 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-09-30 14:00:00 | 343.95 | 2025-10-06 10:15:00 | 351.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-10-01 10:00:00 | 344.50 | 2025-10-06 10:15:00 | 351.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-10-01 12:15:00 | 344.30 | 2025-10-06 10:15:00 | 351.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-10-07 09:15:00 | 355.65 | 2025-10-13 10:15:00 | 361.35 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-10-07 12:45:00 | 353.15 | 2025-10-13 10:15:00 | 361.35 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-10-07 15:15:00 | 353.00 | 2025-10-13 10:15:00 | 361.35 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-10-15 12:00:00 | 362.70 | 2025-10-15 14:15:00 | 371.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-10-15 13:00:00 | 362.60 | 2025-10-15 14:15:00 | 371.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-24 09:15:00 | 378.15 | 2025-10-28 10:15:00 | 368.55 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-10-27 09:45:00 | 374.10 | 2025-10-28 10:15:00 | 368.55 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-27 10:45:00 | 374.15 | 2025-10-28 10:15:00 | 368.55 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-27 14:00:00 | 374.05 | 2025-10-28 10:15:00 | 368.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-11-10 09:30:00 | 365.50 | 2025-11-10 10:15:00 | 374.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-20 11:15:00 | 367.40 | 2025-11-27 09:15:00 | 361.60 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2025-12-26 11:30:00 | 369.50 | 2025-12-29 13:15:00 | 353.90 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2026-01-02 13:30:00 | 364.25 | 2026-01-06 09:15:00 | 354.55 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-01-02 14:15:00 | 364.90 | 2026-01-06 09:15:00 | 354.55 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-01-05 10:00:00 | 363.90 | 2026-01-06 09:15:00 | 354.55 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-14 13:30:00 | 345.80 | 2026-01-16 09:15:00 | 355.35 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-01-14 14:15:00 | 345.60 | 2026-01-16 09:15:00 | 355.35 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-01-16 11:30:00 | 344.00 | 2026-01-19 12:15:00 | 326.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:30:00 | 344.00 | 2026-01-21 09:15:00 | 309.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-02-25 12:30:00 | 272.80 | 2026-02-26 09:15:00 | 277.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-02-26 11:45:00 | 274.70 | 2026-03-02 09:15:00 | 260.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:45:00 | 274.70 | 2026-03-06 10:15:00 | 247.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 225.30 | 2026-03-30 09:15:00 | 214.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 225.30 | 2026-04-01 09:15:00 | 223.01 | STOP_HIT | 0.50 | 1.02% |
| BUY | retest2 | 2026-04-13 10:45:00 | 244.43 | 2026-04-16 09:15:00 | 268.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:15:00 | 243.71 | 2026-04-16 09:15:00 | 268.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 245.44 | 2026-04-16 09:15:00 | 269.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 248.62 | 2026-04-16 09:15:00 | 273.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-20 11:15:00 | 273.95 | 2026-04-22 09:15:00 | 270.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-04-20 12:15:00 | 273.50 | 2026-04-22 09:15:00 | 270.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-04-21 09:30:00 | 274.00 | 2026-04-22 09:15:00 | 270.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-04-21 13:15:00 | 273.68 | 2026-04-22 09:15:00 | 270.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-23 12:15:00 | 270.00 | 2026-04-24 11:15:00 | 256.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 13:30:00 | 269.78 | 2026-04-24 11:15:00 | 256.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 12:15:00 | 270.00 | 2026-04-27 12:15:00 | 259.82 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-04-23 13:30:00 | 269.78 | 2026-04-27 12:15:00 | 259.82 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-05-04 13:15:00 | 259.15 | 2026-05-04 14:15:00 | 261.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-05-05 15:15:00 | 261.60 | 2026-05-08 09:15:00 | 287.76 | TARGET_HIT | 1.00 | 10.00% |
