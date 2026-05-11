# Delhivery Ltd. (DELHIVERY)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 106 |
| ALERT2 | 102 |
| ALERT2_SKIP | 54 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 103 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 80
- **Target hits / Stop hits / Partials:** 4 / 101 / 12
- **Avg / median % per leg:** 0.23% / -0.86%
- **Sum % (uncompounded):** 27.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 11 | 22.0% | 2 | 48 | 0 | -0.38% | -18.9% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.82% | 2.8% |
| BUY @ 3rd Alert (retest2) | 49 | 10 | 20.4% | 2 | 47 | 0 | -0.44% | -21.7% |
| SELL (all) | 67 | 26 | 38.8% | 2 | 53 | 12 | 0.69% | 46.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.38% | -2.4% |
| SELL @ 3rd Alert (retest2) | 66 | 26 | 39.4% | 2 | 52 | 12 | 0.73% | 48.5% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.22% | 0.4% |
| retest2 (combined) | 115 | 36 | 31.3% | 4 | 99 | 12 | 0.23% | 26.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 443.75 | 448.86 | 449.43 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 453.40 | 449.39 | 449.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 455.95 | 450.70 | 449.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 15:15:00 | 452.15 | 452.85 | 451.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 15:15:00 | 452.15 | 452.85 | 451.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 452.15 | 452.85 | 451.42 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 447.00 | 450.83 | 451.02 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 459.40 | 452.18 | 451.30 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 09:15:00 | 440.00 | 451.10 | 451.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-18 11:15:00 | 437.70 | 448.42 | 450.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 402.85 | 395.97 | 408.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:30:00 | 400.00 | 395.97 | 408.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 399.20 | 398.06 | 407.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 398.00 | 398.06 | 407.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 410.00 | 405.30 | 405.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 410.00 | 405.30 | 405.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 14:15:00 | 414.45 | 407.87 | 406.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 410.15 | 413.87 | 411.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 410.15 | 413.87 | 411.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 410.15 | 413.87 | 411.44 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 404.75 | 409.28 | 409.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 397.50 | 405.12 | 407.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 392.45 | 390.14 | 394.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 392.45 | 390.14 | 394.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 392.45 | 390.14 | 394.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 362.00 | 391.12 | 393.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 11:15:00 | 383.30 | 379.91 | 379.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 383.30 | 379.91 | 379.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 387.90 | 383.10 | 381.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 385.00 | 385.13 | 383.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:15:00 | 386.05 | 385.13 | 383.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 389.45 | 389.64 | 387.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 387.75 | 389.64 | 387.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 386.70 | 389.10 | 388.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 386.70 | 389.10 | 388.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 386.95 | 388.67 | 387.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 390.00 | 388.67 | 387.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 399.05 | 400.94 | 401.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 399.05 | 400.94 | 401.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 397.15 | 400.18 | 400.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 394.65 | 394.08 | 396.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 394.65 | 394.08 | 396.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 394.65 | 394.08 | 396.26 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 398.00 | 396.56 | 396.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 11:15:00 | 400.15 | 399.53 | 398.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 12:15:00 | 399.40 | 400.08 | 399.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 12:15:00 | 399.40 | 400.08 | 399.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 399.40 | 400.08 | 399.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 399.40 | 400.08 | 399.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 400.00 | 400.06 | 399.51 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 397.00 | 398.90 | 399.15 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 399.95 | 399.13 | 399.03 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 397.75 | 398.74 | 398.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 12:15:00 | 396.85 | 398.27 | 398.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 398.05 | 397.90 | 398.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 398.05 | 397.90 | 398.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 398.05 | 397.90 | 398.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 398.05 | 397.90 | 398.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 399.25 | 398.17 | 398.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 396.65 | 398.17 | 398.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 396.30 | 397.79 | 398.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:00:00 | 395.05 | 396.77 | 397.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 394.00 | 393.34 | 394.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:15:00 | 375.30 | 379.51 | 384.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-12 13:15:00 | 379.15 | 378.82 | 382.23 | SL hit (close>ema200) qty=0.50 sl=378.82 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 383.20 | 380.31 | 380.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 385.45 | 382.44 | 381.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 381.05 | 382.16 | 381.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 381.05 | 382.16 | 381.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 381.05 | 382.16 | 381.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 381.05 | 382.16 | 381.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 385.20 | 382.77 | 381.70 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 374.00 | 381.08 | 381.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 370.60 | 375.13 | 377.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 375.80 | 374.16 | 376.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 375.80 | 374.16 | 376.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 375.80 | 374.16 | 376.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 377.40 | 374.16 | 376.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 377.10 | 374.75 | 376.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 377.10 | 374.75 | 376.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 378.45 | 375.49 | 376.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 378.45 | 375.49 | 376.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 382.50 | 377.29 | 377.39 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 383.35 | 378.50 | 377.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 384.90 | 380.45 | 379.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 382.90 | 383.26 | 381.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 15:00:00 | 382.90 | 383.26 | 381.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 382.80 | 383.17 | 381.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 382.45 | 383.17 | 381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 382.25 | 382.99 | 381.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:30:00 | 386.50 | 383.92 | 382.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:30:00 | 386.15 | 383.50 | 382.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-05 09:15:00 | 425.15 | 414.37 | 410.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 15:15:00 | 406.00 | 409.40 | 409.43 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 412.90 | 410.10 | 409.75 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 15:15:00 | 405.30 | 409.30 | 409.75 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 411.65 | 409.86 | 409.62 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 405.60 | 408.88 | 409.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 404.05 | 407.23 | 408.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 09:15:00 | 405.10 | 404.56 | 406.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 405.10 | 404.56 | 406.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 405.10 | 404.56 | 406.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 405.10 | 404.56 | 406.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 408.50 | 405.35 | 406.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:45:00 | 409.55 | 405.35 | 406.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 408.75 | 406.03 | 407.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 13:45:00 | 407.20 | 406.69 | 407.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:15:00 | 406.65 | 406.69 | 407.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:45:00 | 405.50 | 406.29 | 406.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:00:00 | 407.05 | 406.26 | 406.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 407.20 | 406.73 | 406.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:30:00 | 408.00 | 406.73 | 406.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 408.05 | 406.99 | 406.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 14:15:00 | 408.05 | 406.99 | 406.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 15:15:00 | 408.80 | 407.35 | 407.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 405.50 | 406.98 | 406.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 405.50 | 406.98 | 406.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 405.50 | 406.98 | 406.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 401.80 | 406.98 | 406.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 410.90 | 407.77 | 407.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:45:00 | 412.90 | 409.54 | 408.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 413.05 | 412.54 | 410.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 14:15:00 | 422.80 | 428.29 | 428.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 422.80 | 428.29 | 428.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 15:15:00 | 419.90 | 426.61 | 428.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 426.40 | 425.86 | 427.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 426.40 | 425.86 | 427.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 426.40 | 425.86 | 427.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 428.35 | 425.86 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 426.85 | 426.06 | 427.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 427.05 | 426.06 | 427.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 427.80 | 426.41 | 427.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 427.80 | 426.41 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 427.55 | 426.63 | 427.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 427.05 | 426.63 | 427.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 429.00 | 427.11 | 427.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 423.25 | 427.11 | 427.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 422.85 | 426.26 | 427.08 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 431.00 | 427.34 | 427.29 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 423.65 | 427.87 | 428.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 422.35 | 426.77 | 427.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 421.20 | 419.70 | 422.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:00:00 | 421.20 | 419.70 | 422.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 419.70 | 419.70 | 421.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 417.60 | 419.28 | 421.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 417.40 | 418.81 | 420.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 424.35 | 419.82 | 419.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 424.35 | 419.82 | 419.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 425.00 | 421.93 | 420.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 421.70 | 422.49 | 421.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:00:00 | 421.70 | 422.49 | 421.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 423.00 | 422.59 | 421.65 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 420.55 | 421.19 | 421.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 417.05 | 419.12 | 420.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 410.50 | 409.83 | 413.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 410.50 | 409.83 | 413.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 413.95 | 410.41 | 412.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 413.45 | 410.41 | 412.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 417.30 | 411.79 | 413.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 417.30 | 411.79 | 413.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 414.70 | 413.92 | 413.84 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 10:15:00 | 413.45 | 413.83 | 413.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 410.00 | 412.77 | 413.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 413.20 | 412.45 | 413.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 413.20 | 412.45 | 413.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 413.20 | 412.45 | 413.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 413.65 | 412.45 | 413.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 415.40 | 413.04 | 413.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 415.40 | 413.04 | 413.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 416.95 | 413.82 | 413.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 418.05 | 415.33 | 414.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 415.25 | 415.74 | 414.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:00:00 | 415.25 | 415.74 | 414.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 415.70 | 415.73 | 414.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:45:00 | 415.00 | 415.73 | 414.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 415.10 | 415.66 | 414.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:30:00 | 414.55 | 415.66 | 414.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 414.35 | 415.40 | 414.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:30:00 | 414.35 | 415.40 | 414.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 413.90 | 415.10 | 414.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 413.90 | 415.10 | 414.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 414.75 | 415.03 | 414.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 419.25 | 415.03 | 414.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 431.30 | 433.02 | 433.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 431.30 | 433.02 | 433.05 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 433.50 | 433.11 | 433.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 435.00 | 433.49 | 433.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 10:15:00 | 430.55 | 434.32 | 433.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 430.55 | 434.32 | 433.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 430.55 | 434.32 | 433.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 430.55 | 434.32 | 433.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 430.95 | 433.65 | 433.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:15:00 | 430.10 | 433.65 | 433.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 429.70 | 432.86 | 433.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 423.90 | 430.04 | 431.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 423.85 | 422.93 | 425.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 423.85 | 422.93 | 425.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 423.85 | 422.93 | 425.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:30:00 | 425.75 | 422.93 | 425.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 408.20 | 406.00 | 410.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 406.50 | 406.00 | 410.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 415.00 | 407.80 | 410.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 415.00 | 407.80 | 410.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 411.05 | 408.45 | 410.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 410.00 | 408.45 | 410.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 415.45 | 412.16 | 411.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 415.45 | 412.16 | 411.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 424.05 | 414.54 | 413.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 12:15:00 | 414.25 | 416.45 | 414.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 12:15:00 | 414.25 | 416.45 | 414.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 414.25 | 416.45 | 414.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:00:00 | 414.25 | 416.45 | 414.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 415.45 | 416.25 | 414.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:30:00 | 415.95 | 416.25 | 414.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 414.65 | 415.93 | 414.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:45:00 | 414.65 | 415.93 | 414.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 413.50 | 415.44 | 414.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 413.00 | 415.44 | 414.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 411.60 | 414.67 | 414.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 411.40 | 414.67 | 414.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 409.70 | 413.68 | 413.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 409.20 | 411.76 | 412.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 411.10 | 410.16 | 411.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 411.10 | 410.16 | 411.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 411.10 | 410.16 | 411.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 411.10 | 410.16 | 411.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 410.00 | 410.12 | 411.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 412.75 | 410.12 | 411.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 413.10 | 410.72 | 411.44 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 414.85 | 412.08 | 411.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 12:15:00 | 418.45 | 413.35 | 412.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 410.00 | 414.40 | 413.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 410.00 | 414.40 | 413.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 410.00 | 414.40 | 413.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:45:00 | 410.35 | 414.40 | 413.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 410.35 | 413.59 | 413.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 410.55 | 413.59 | 413.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 409.75 | 412.55 | 412.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 403.90 | 409.84 | 411.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 12:15:00 | 407.40 | 407.21 | 409.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 12:45:00 | 407.25 | 407.21 | 409.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 408.80 | 407.53 | 409.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:45:00 | 408.55 | 407.53 | 409.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 410.00 | 408.02 | 409.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 409.90 | 408.02 | 409.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 405.70 | 407.56 | 409.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 403.00 | 406.28 | 408.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 382.85 | 390.06 | 394.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 09:15:00 | 362.70 | 367.61 | 373.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 364.30 | 357.50 | 357.00 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 353.95 | 357.15 | 357.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 351.95 | 354.60 | 355.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 352.50 | 352.18 | 354.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 352.50 | 352.18 | 354.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 354.20 | 352.58 | 354.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 353.65 | 352.58 | 354.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 353.50 | 352.77 | 354.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 356.30 | 352.77 | 354.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 356.55 | 353.52 | 354.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 12:15:00 | 353.30 | 354.19 | 354.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 356.50 | 354.91 | 354.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 356.50 | 354.91 | 354.76 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 353.75 | 354.64 | 354.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 351.50 | 353.36 | 354.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 340.75 | 331.47 | 332.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 340.75 | 331.47 | 332.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 340.75 | 331.47 | 332.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 340.25 | 331.47 | 332.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 343.40 | 333.86 | 333.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 350.05 | 337.10 | 335.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 343.75 | 344.56 | 342.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 343.75 | 344.56 | 342.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 343.75 | 344.56 | 342.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 343.95 | 344.56 | 342.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 341.60 | 346.51 | 345.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 341.60 | 346.51 | 345.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 339.10 | 345.02 | 344.70 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 337.65 | 343.55 | 344.06 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 348.90 | 343.11 | 342.40 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 339.25 | 342.95 | 343.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 335.05 | 340.50 | 342.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 337.65 | 334.72 | 336.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 337.65 | 334.72 | 336.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 337.65 | 334.72 | 336.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 337.65 | 334.72 | 336.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 337.85 | 335.35 | 336.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 339.90 | 335.35 | 336.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 338.00 | 335.88 | 336.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 338.00 | 335.88 | 336.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 341.30 | 338.18 | 337.78 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 333.45 | 337.68 | 338.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 332.60 | 334.73 | 336.28 | Break + close below crossover candle low |

### Cycle 48 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 356.45 | 338.90 | 337.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 359.55 | 351.28 | 345.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 389.75 | 392.54 | 389.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 13:15:00 | 389.75 | 392.54 | 389.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 389.75 | 392.54 | 389.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 389.75 | 392.54 | 389.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 383.70 | 390.78 | 388.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 383.70 | 390.78 | 388.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 386.00 | 389.82 | 388.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 386.25 | 388.71 | 388.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 383.75 | 387.72 | 387.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 383.75 | 387.72 | 387.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 376.30 | 385.43 | 386.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 355.40 | 355.27 | 362.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 355.40 | 355.27 | 362.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 365.30 | 359.19 | 361.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 365.30 | 359.19 | 361.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 360.50 | 359.45 | 361.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 365.15 | 359.45 | 361.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 353.00 | 358.16 | 360.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:00:00 | 350.05 | 356.54 | 359.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:45:00 | 350.25 | 353.80 | 356.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 351.35 | 350.73 | 353.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:00:00 | 350.70 | 350.47 | 352.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 349.45 | 347.56 | 349.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 349.45 | 347.56 | 349.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 346.00 | 347.25 | 349.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 345.20 | 347.25 | 349.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 343.25 | 346.45 | 348.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 345.10 | 345.58 | 347.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 350.60 | 347.08 | 347.67 | SL hit (close>static) qty=1.00 sl=349.90 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 349.05 | 347.90 | 347.80 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 344.60 | 347.69 | 347.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 342.70 | 346.70 | 347.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 338.00 | 337.77 | 340.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:30:00 | 338.00 | 337.77 | 340.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 338.80 | 336.48 | 337.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 338.80 | 336.48 | 337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 340.50 | 337.29 | 338.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 341.00 | 337.29 | 338.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 337.80 | 337.49 | 337.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:30:00 | 336.75 | 337.54 | 337.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 336.70 | 337.35 | 337.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 339.70 | 337.12 | 337.54 | SL hit (close>static) qty=1.00 sl=338.50 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 332.90 | 327.50 | 326.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 337.00 | 331.50 | 329.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 336.90 | 340.32 | 337.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 336.90 | 340.32 | 337.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 336.90 | 340.32 | 337.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 336.90 | 340.32 | 337.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 339.10 | 340.07 | 337.52 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 329.60 | 336.22 | 336.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 328.80 | 334.74 | 335.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 322.95 | 322.68 | 327.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 324.25 | 322.68 | 327.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 325.90 | 323.97 | 327.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:00:00 | 324.05 | 324.41 | 326.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:00:00 | 325.55 | 325.50 | 326.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 325.25 | 326.53 | 326.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 11:15:00 | 309.27 | 318.30 | 322.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 308.99 | 315.55 | 319.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 15:15:00 | 315.60 | 315.56 | 319.57 | SL hit (close>ema200) qty=0.50 sl=315.56 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 322.90 | 318.72 | 318.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 324.15 | 319.81 | 318.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 317.90 | 321.50 | 320.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 317.90 | 321.50 | 320.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 317.90 | 321.50 | 320.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 317.90 | 321.50 | 320.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 323.60 | 321.92 | 320.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 319.60 | 321.92 | 320.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 319.50 | 321.44 | 320.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 320.15 | 321.44 | 320.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 321.00 | 321.35 | 320.59 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 319.00 | 320.21 | 320.22 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 320.95 | 320.36 | 320.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 15:15:00 | 322.00 | 320.69 | 320.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 317.75 | 320.67 | 320.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 317.75 | 320.67 | 320.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 317.75 | 320.67 | 320.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 316.55 | 320.67 | 320.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 316.60 | 319.86 | 320.20 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 322.65 | 320.73 | 320.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 10:15:00 | 329.90 | 322.56 | 321.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 13:15:00 | 323.00 | 323.24 | 322.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:00:00 | 323.00 | 323.24 | 322.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 323.25 | 323.24 | 322.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:30:00 | 322.50 | 323.24 | 322.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 323.00 | 323.19 | 322.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 327.30 | 323.19 | 322.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 322.10 | 322.83 | 322.21 | SL hit (close<static) qty=1.00 sl=322.20 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 321.95 | 324.55 | 324.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 314.10 | 321.25 | 323.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 15:15:00 | 322.00 | 316.03 | 318.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 15:15:00 | 322.00 | 316.03 | 318.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 322.00 | 316.03 | 318.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 303.35 | 316.03 | 318.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:15:00 | 288.18 | 297.33 | 305.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 273.02 | 285.77 | 295.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 275.45 | 271.22 | 270.66 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 272.50 | 274.05 | 274.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 266.25 | 272.10 | 273.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 249.28 | 249.18 | 253.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:00:00 | 249.28 | 249.18 | 253.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 251.25 | 250.01 | 253.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 249.33 | 250.47 | 252.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 252.81 | 251.69 | 251.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 252.81 | 251.69 | 251.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 255.66 | 252.49 | 252.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 257.91 | 259.89 | 257.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 257.91 | 259.89 | 257.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 257.91 | 259.89 | 257.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 257.91 | 259.89 | 257.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 259.05 | 259.72 | 257.85 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 253.26 | 257.01 | 257.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 250.88 | 254.36 | 255.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 11:15:00 | 242.67 | 241.91 | 245.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 12:00:00 | 242.67 | 241.91 | 245.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 247.99 | 243.11 | 244.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:00:00 | 247.99 | 243.11 | 244.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 246.20 | 243.73 | 244.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:45:00 | 245.25 | 244.26 | 244.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:00:00 | 244.79 | 244.36 | 244.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 249.24 | 244.87 | 244.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 249.24 | 244.87 | 244.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 249.62 | 246.46 | 245.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 256.31 | 257.25 | 253.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 256.31 | 257.25 | 253.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 260.11 | 262.16 | 260.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 260.11 | 262.16 | 260.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 259.75 | 261.68 | 260.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 258.60 | 261.68 | 260.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 252.36 | 258.70 | 259.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 251.09 | 254.63 | 256.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 253.36 | 252.97 | 254.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 253.36 | 252.97 | 254.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 253.28 | 253.03 | 254.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 258.95 | 253.03 | 254.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 260.30 | 254.48 | 255.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 260.07 | 254.48 | 255.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 258.23 | 255.23 | 255.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 257.06 | 255.23 | 255.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 258.54 | 255.89 | 255.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 258.54 | 255.89 | 255.68 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 253.60 | 255.20 | 255.42 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 260.75 | 256.31 | 255.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 264.15 | 261.13 | 259.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 251.05 | 260.28 | 259.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 251.05 | 260.28 | 259.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 251.05 | 260.28 | 259.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 251.05 | 260.28 | 259.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 256.00 | 259.42 | 259.55 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 11:15:00 | 260.70 | 258.53 | 258.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 13:15:00 | 267.40 | 260.91 | 259.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-08 09:15:00 | 253.20 | 261.61 | 260.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 253.20 | 261.61 | 260.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 253.20 | 261.61 | 260.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 253.20 | 261.61 | 260.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 10:15:00 | 249.35 | 259.15 | 259.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 240.65 | 250.51 | 254.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 14:15:00 | 246.95 | 245.21 | 247.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-11 15:00:00 | 246.95 | 245.21 | 247.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 253.10 | 247.06 | 248.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 253.10 | 247.06 | 248.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 253.85 | 248.42 | 248.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 253.85 | 248.42 | 248.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 255.00 | 249.73 | 249.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 255.75 | 250.94 | 249.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 10:15:00 | 298.10 | 298.86 | 294.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 11:00:00 | 298.10 | 298.86 | 294.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 308.05 | 303.95 | 300.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 310.20 | 304.70 | 302.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:45:00 | 310.00 | 305.80 | 303.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 10:45:00 | 309.15 | 306.46 | 303.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 305.05 | 306.24 | 306.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 11:15:00 | 305.05 | 306.24 | 306.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 302.30 | 304.89 | 305.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 307.00 | 304.71 | 305.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 307.00 | 304.71 | 305.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 307.00 | 304.71 | 305.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 307.00 | 304.71 | 305.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 306.60 | 305.09 | 305.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:30:00 | 303.70 | 304.81 | 305.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:15:00 | 304.00 | 304.81 | 305.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 302.95 | 304.69 | 305.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:00:00 | 303.90 | 304.59 | 304.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 301.70 | 303.26 | 304.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:30:00 | 303.35 | 303.26 | 304.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 300.30 | 301.66 | 303.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 306.50 | 303.11 | 303.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 306.50 | 303.11 | 303.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 307.00 | 303.89 | 303.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 322.80 | 323.38 | 320.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 322.80 | 323.38 | 320.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 321.20 | 322.95 | 320.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 321.20 | 322.95 | 320.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 320.70 | 322.50 | 320.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:00:00 | 320.70 | 322.50 | 320.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 320.85 | 322.17 | 320.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 343.10 | 321.53 | 320.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 358.25 | 360.00 | 360.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 358.25 | 360.00 | 360.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 357.40 | 359.48 | 359.77 | Break + close below crossover candle low |

### Cycle 76 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 365.00 | 360.43 | 360.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 367.50 | 361.85 | 360.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 368.15 | 368.94 | 366.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 365.65 | 368.94 | 366.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 364.30 | 368.01 | 366.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 364.50 | 368.01 | 366.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 365.75 | 367.56 | 366.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 367.70 | 367.09 | 366.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 367.75 | 367.27 | 366.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:30:00 | 367.70 | 368.39 | 368.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 366.65 | 368.04 | 368.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 366.65 | 368.04 | 368.15 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 372.00 | 368.52 | 368.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 373.20 | 371.15 | 369.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 370.50 | 371.10 | 370.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 370.50 | 371.10 | 370.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 370.50 | 371.10 | 370.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 369.95 | 371.10 | 370.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 369.75 | 370.83 | 370.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 368.85 | 370.83 | 370.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 367.60 | 370.19 | 369.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 367.60 | 370.19 | 369.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 367.10 | 369.57 | 369.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 366.00 | 368.43 | 369.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 368.60 | 368.46 | 369.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 368.60 | 368.46 | 369.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 368.60 | 368.46 | 369.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 369.20 | 368.46 | 369.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 366.50 | 368.07 | 368.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:45:00 | 365.45 | 367.45 | 368.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 365.00 | 366.87 | 367.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:45:00 | 363.80 | 366.13 | 367.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 347.18 | 360.46 | 363.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 346.75 | 360.46 | 363.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 345.61 | 360.46 | 363.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 360.95 | 358.86 | 361.37 | SL hit (close>ema200) qty=0.50 sl=358.86 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 365.55 | 361.71 | 361.60 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 360.55 | 361.57 | 361.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 359.35 | 361.12 | 361.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 365.70 | 361.45 | 361.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 365.70 | 361.45 | 361.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 365.70 | 361.45 | 361.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 365.70 | 361.45 | 361.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 365.45 | 362.25 | 361.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 367.55 | 363.31 | 362.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 359.05 | 363.50 | 363.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 359.05 | 363.50 | 363.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 359.05 | 363.50 | 363.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 359.05 | 363.50 | 363.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 361.85 | 363.17 | 362.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 360.70 | 363.17 | 362.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 360.40 | 362.62 | 362.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 357.35 | 361.56 | 362.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 354.85 | 354.82 | 357.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:45:00 | 355.00 | 354.82 | 357.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 357.40 | 355.34 | 357.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 357.40 | 355.34 | 357.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 357.95 | 355.86 | 357.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 357.30 | 355.86 | 357.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 359.00 | 356.49 | 357.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 360.60 | 356.49 | 357.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 358.25 | 356.84 | 357.81 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 360.70 | 358.66 | 358.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 372.00 | 361.59 | 359.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 383.50 | 385.04 | 379.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:45:00 | 383.55 | 385.04 | 379.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 383.55 | 384.43 | 382.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 383.55 | 384.43 | 382.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 382.75 | 384.09 | 382.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 384.55 | 384.09 | 382.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 383.95 | 384.07 | 382.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 385.75 | 384.02 | 382.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 381.35 | 383.14 | 382.81 | SL hit (close<static) qty=1.00 sl=382.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 379.70 | 382.54 | 382.70 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 13:15:00 | 388.60 | 383.77 | 383.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 391.50 | 386.00 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 385.25 | 385.92 | 384.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:30:00 | 385.00 | 385.92 | 384.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 388.20 | 386.37 | 385.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 387.80 | 386.37 | 385.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 412.40 | 415.76 | 412.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 411.35 | 415.76 | 412.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 410.55 | 414.71 | 412.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 410.55 | 414.71 | 412.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 409.55 | 413.68 | 411.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 409.55 | 413.68 | 411.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 412.20 | 412.41 | 411.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:15:00 | 409.45 | 412.41 | 411.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 409.45 | 411.82 | 411.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 408.15 | 411.82 | 411.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 409.30 | 411.32 | 411.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 406.20 | 410.29 | 410.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 412.25 | 408.68 | 409.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 412.25 | 408.68 | 409.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 412.25 | 408.68 | 409.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 411.25 | 408.68 | 409.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 413.45 | 409.64 | 409.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 413.45 | 409.64 | 409.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 415.10 | 410.73 | 410.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 416.95 | 411.97 | 410.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 426.85 | 427.29 | 423.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 13:30:00 | 427.70 | 427.29 | 423.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 425.20 | 426.46 | 424.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 428.25 | 426.46 | 424.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 431.00 | 436.78 | 437.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 431.00 | 436.78 | 437.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 428.55 | 435.13 | 436.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 10:15:00 | 419.80 | 414.15 | 418.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 10:15:00 | 419.80 | 414.15 | 418.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 419.80 | 414.15 | 418.24 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 14:15:00 | 424.95 | 421.13 | 420.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 432.60 | 424.34 | 422.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 429.05 | 429.60 | 426.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:15:00 | 450.65 | 429.60 | 426.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 463.35 | 466.17 | 463.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 463.35 | 466.17 | 463.84 | SL hit (close<ema400) qty=1.00 sl=463.84 alert=retest1 |

### Cycle 91 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 464.50 | 464.91 | 464.96 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 465.95 | 465.12 | 465.05 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 463.30 | 464.75 | 464.89 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 470.70 | 465.85 | 465.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 471.70 | 469.01 | 467.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 471.60 | 472.01 | 469.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 471.60 | 472.01 | 469.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 469.35 | 471.24 | 469.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:15:00 | 473.30 | 471.31 | 470.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 473.85 | 470.99 | 470.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 473.00 | 471.19 | 470.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 475.15 | 471.19 | 470.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 472.50 | 472.76 | 471.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 476.90 | 473.46 | 472.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:45:00 | 474.95 | 474.86 | 473.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:30:00 | 475.35 | 475.15 | 474.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:45:00 | 476.35 | 476.54 | 475.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 476.25 | 476.48 | 475.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 475.75 | 476.48 | 475.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 473.95 | 475.97 | 475.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 473.95 | 475.97 | 475.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 471.50 | 475.08 | 474.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 471.50 | 475.08 | 474.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 468.95 | 473.55 | 474.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 468.95 | 473.55 | 474.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 466.00 | 471.47 | 473.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 470.00 | 469.58 | 471.27 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 09:15:00 | 464.00 | 469.58 | 471.27 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 471.00 | 469.87 | 471.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 471.35 | 469.87 | 471.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 475.05 | 470.90 | 471.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 475.05 | 470.90 | 471.59 | SL hit (close>ema400) qty=1.00 sl=471.59 alert=retest1 |

### Cycle 96 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 474.20 | 472.38 | 472.20 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 471.15 | 472.26 | 472.36 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 485.10 | 474.90 | 473.54 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 470.45 | 474.95 | 475.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 467.15 | 472.75 | 474.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 470.00 | 469.01 | 471.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 465.70 | 469.01 | 471.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 473.55 | 469.94 | 471.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 473.55 | 469.94 | 471.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 474.25 | 470.80 | 471.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 474.25 | 470.80 | 471.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 474.30 | 472.41 | 472.21 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 468.05 | 471.88 | 472.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 463.00 | 470.10 | 471.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 465.05 | 464.74 | 467.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 12:00:00 | 465.05 | 464.74 | 467.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 466.45 | 465.09 | 467.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 467.05 | 465.09 | 467.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 463.85 | 464.84 | 466.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 463.45 | 464.84 | 466.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 467.20 | 465.52 | 466.43 | SL hit (close>static) qty=1.00 sl=467.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 468.30 | 467.13 | 467.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 470.00 | 467.71 | 467.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 468.25 | 468.37 | 467.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 468.25 | 468.37 | 467.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 468.25 | 468.37 | 467.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 472.75 | 468.81 | 468.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 471.60 | 470.07 | 469.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 471.30 | 470.17 | 469.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 472.55 | 478.14 | 478.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 472.55 | 478.14 | 478.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 470.65 | 474.27 | 476.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 459.70 | 459.02 | 464.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:00:00 | 459.70 | 459.02 | 464.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 461.00 | 459.41 | 463.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 462.05 | 459.41 | 463.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 459.35 | 458.31 | 460.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 460.45 | 458.31 | 460.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 452.10 | 445.54 | 449.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 452.10 | 445.54 | 449.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 449.60 | 446.35 | 449.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 452.25 | 446.35 | 449.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 449.95 | 447.07 | 449.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 449.95 | 447.07 | 449.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 448.95 | 447.45 | 449.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 447.60 | 447.51 | 449.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 453.90 | 449.19 | 449.54 | SL hit (close>static) qty=1.00 sl=450.10 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 456.50 | 443.22 | 441.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 461.90 | 450.72 | 445.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 462.20 | 464.41 | 458.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 462.20 | 464.41 | 458.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 466.20 | 467.46 | 465.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 466.20 | 467.46 | 465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 463.70 | 466.70 | 465.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 463.70 | 466.70 | 465.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 464.10 | 466.18 | 465.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 463.50 | 466.18 | 465.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 466.85 | 466.04 | 465.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 468.30 | 466.04 | 465.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 472.10 | 466.85 | 465.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 468.55 | 469.66 | 467.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 467.95 | 468.15 | 467.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 467.00 | 468.17 | 467.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 462.00 | 466.93 | 467.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 462.00 | 466.93 | 467.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 11:15:00 | 459.80 | 465.51 | 466.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 454.25 | 452.78 | 457.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 454.25 | 452.78 | 457.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 459.35 | 454.71 | 457.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 459.35 | 454.71 | 457.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 462.05 | 456.18 | 458.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 462.00 | 456.18 | 458.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 465.70 | 459.27 | 459.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 470.45 | 462.63 | 460.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 10:15:00 | 474.05 | 474.63 | 470.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 11:00:00 | 474.05 | 474.63 | 470.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 466.80 | 472.99 | 471.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 466.80 | 472.99 | 471.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 463.85 | 471.16 | 470.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 463.55 | 471.16 | 470.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 466.30 | 470.19 | 470.30 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 473.10 | 470.06 | 469.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 478.05 | 472.86 | 471.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 473.05 | 478.83 | 476.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 473.05 | 478.83 | 476.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 473.05 | 478.83 | 476.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 473.05 | 478.83 | 476.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 477.05 | 478.47 | 476.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 479.70 | 476.24 | 475.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 472.70 | 475.47 | 475.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 472.70 | 475.47 | 475.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 470.50 | 473.85 | 474.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 472.60 | 470.69 | 472.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 472.60 | 470.69 | 472.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 472.60 | 470.69 | 472.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 472.60 | 470.69 | 472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 471.25 | 470.80 | 472.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:30:00 | 472.50 | 470.80 | 472.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 472.40 | 471.12 | 472.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 472.40 | 471.12 | 472.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 472.00 | 471.30 | 472.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:15:00 | 472.00 | 471.30 | 472.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 472.00 | 471.44 | 472.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 475.55 | 471.44 | 472.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 484.00 | 473.95 | 473.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 10:15:00 | 485.80 | 476.32 | 474.50 | Break + close above crossover candle high |

### Cycle 111 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 450.30 | 475.85 | 475.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 436.40 | 453.71 | 463.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 10:15:00 | 430.10 | 429.73 | 437.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 10:45:00 | 433.80 | 429.73 | 437.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 432.40 | 428.35 | 431.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 432.40 | 428.35 | 431.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 432.35 | 429.15 | 431.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 433.45 | 429.15 | 431.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 430.95 | 429.51 | 431.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:15:00 | 432.00 | 429.51 | 431.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 432.00 | 430.01 | 431.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 429.00 | 430.01 | 431.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 434.50 | 430.91 | 432.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 434.50 | 430.91 | 432.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 436.80 | 432.08 | 432.54 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 440.55 | 433.78 | 433.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 442.80 | 438.03 | 436.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 431.25 | 437.86 | 437.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 431.25 | 437.86 | 437.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 431.25 | 437.86 | 437.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 431.25 | 437.86 | 437.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 428.50 | 435.99 | 436.38 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 440.75 | 436.43 | 436.21 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 431.10 | 435.78 | 436.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 430.35 | 434.69 | 435.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 409.20 | 409.18 | 414.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:45:00 | 409.30 | 409.18 | 414.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 414.00 | 410.46 | 413.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 419.25 | 410.46 | 413.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 415.60 | 411.49 | 413.95 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 418.90 | 415.36 | 415.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 419.75 | 416.24 | 415.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 422.30 | 425.46 | 423.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 422.30 | 425.46 | 423.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 422.30 | 425.46 | 423.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 422.30 | 425.46 | 423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 424.40 | 425.25 | 423.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 422.25 | 425.25 | 423.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 421.70 | 424.54 | 423.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 421.20 | 424.54 | 423.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 417.80 | 423.19 | 422.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 417.80 | 423.19 | 422.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 418.95 | 422.34 | 422.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 416.00 | 421.07 | 421.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 402.55 | 402.04 | 406.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 402.55 | 402.04 | 406.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 404.65 | 402.86 | 405.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 404.60 | 402.86 | 405.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 405.70 | 403.43 | 405.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 405.70 | 403.43 | 405.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 406.65 | 404.07 | 405.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 403.15 | 404.20 | 405.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 403.60 | 404.15 | 405.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 404.30 | 402.19 | 402.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 411.35 | 404.02 | 403.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 411.35 | 404.02 | 403.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 413.65 | 405.95 | 404.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 408.55 | 410.32 | 407.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 408.55 | 410.32 | 407.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 408.55 | 410.32 | 407.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 408.55 | 410.32 | 407.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 408.60 | 409.97 | 407.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:30:00 | 407.65 | 409.97 | 407.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 407.85 | 409.55 | 407.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 407.40 | 409.55 | 407.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 403.35 | 408.31 | 407.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 403.35 | 408.31 | 407.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 405.05 | 407.66 | 407.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 401.45 | 407.66 | 407.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 406.30 | 406.95 | 406.99 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 408.30 | 407.22 | 407.11 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 405.00 | 406.78 | 406.91 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 408.05 | 407.03 | 407.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 410.05 | 407.63 | 407.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 408.20 | 410.80 | 409.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 408.20 | 410.80 | 409.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 408.20 | 410.80 | 409.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 408.20 | 410.80 | 409.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 407.85 | 410.21 | 409.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 407.85 | 410.21 | 409.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 408.85 | 409.94 | 409.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 408.00 | 409.94 | 409.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 407.50 | 409.12 | 409.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 404.45 | 407.86 | 408.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 401.00 | 399.70 | 401.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 401.00 | 399.70 | 401.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 403.90 | 400.54 | 402.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 403.90 | 400.54 | 402.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 400.55 | 400.54 | 401.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 400.00 | 400.54 | 401.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 403.25 | 402.43 | 402.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 403.25 | 402.43 | 402.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 406.05 | 403.55 | 402.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 411.55 | 411.66 | 409.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 411.55 | 411.66 | 409.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 409.65 | 411.26 | 409.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 409.65 | 411.26 | 409.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 410.60 | 411.13 | 409.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 409.80 | 411.13 | 409.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 409.95 | 410.89 | 409.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 409.90 | 410.89 | 409.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 406.25 | 409.96 | 409.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 406.45 | 409.96 | 409.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 407.25 | 409.42 | 409.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 408.85 | 409.42 | 409.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 406.55 | 408.85 | 409.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 406.55 | 408.85 | 409.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 404.40 | 407.01 | 407.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 405.00 | 403.22 | 404.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 405.00 | 403.22 | 404.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 405.00 | 403.22 | 404.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 405.00 | 403.22 | 404.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 405.70 | 403.71 | 404.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 405.70 | 403.71 | 404.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 403.30 | 403.63 | 404.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 402.65 | 403.62 | 404.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 402.65 | 403.48 | 404.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 402.85 | 402.07 | 402.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 404.25 | 403.03 | 402.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 404.25 | 403.03 | 402.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 405.55 | 403.72 | 403.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 418.30 | 418.77 | 415.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:30:00 | 419.95 | 418.77 | 415.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 415.25 | 417.67 | 415.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 418.40 | 417.67 | 415.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 415.50 | 417.23 | 415.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:45:00 | 416.80 | 416.93 | 415.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 416.85 | 417.12 | 415.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 411.75 | 415.54 | 415.15 | SL hit (close<static) qty=1.00 sl=412.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 409.60 | 414.35 | 414.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 407.25 | 412.93 | 413.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 396.95 | 396.22 | 402.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 391.00 | 396.22 | 402.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 396.45 | 395.14 | 398.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 398.20 | 395.14 | 398.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 394.80 | 395.07 | 398.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 398.15 | 395.07 | 398.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 400.65 | 396.18 | 398.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 400.65 | 396.18 | 398.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 402.05 | 397.36 | 398.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 402.05 | 397.36 | 398.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 403.50 | 399.80 | 399.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 407.20 | 401.74 | 400.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 401.75 | 403.42 | 401.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 401.75 | 403.42 | 401.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 401.75 | 403.42 | 401.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 401.75 | 403.42 | 401.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 401.50 | 403.04 | 401.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 402.30 | 403.04 | 401.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 401.00 | 402.63 | 401.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 401.15 | 402.63 | 401.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 401.50 | 402.42 | 401.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 401.30 | 402.42 | 401.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 401.80 | 402.30 | 401.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 402.35 | 402.30 | 401.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 401.30 | 402.10 | 401.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:30:00 | 400.80 | 402.10 | 401.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 401.00 | 401.88 | 401.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 401.00 | 401.88 | 401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 401.50 | 401.80 | 401.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:15:00 | 401.00 | 401.80 | 401.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 401.00 | 401.64 | 401.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 393.75 | 400.06 | 400.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 388.15 | 382.87 | 387.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 388.15 | 382.87 | 387.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 388.15 | 382.87 | 387.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 387.25 | 382.87 | 387.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 381.65 | 382.62 | 386.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 380.75 | 382.62 | 386.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 390.10 | 384.74 | 387.13 | SL hit (close>static) qty=1.00 sl=389.70 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 398.75 | 389.49 | 388.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 403.45 | 397.42 | 394.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 401.50 | 402.64 | 398.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 401.50 | 402.64 | 398.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 440.00 | 426.75 | 418.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 425.95 | 426.75 | 418.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 441.80 | 445.38 | 442.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 441.80 | 445.38 | 442.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 441.60 | 444.62 | 442.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 438.95 | 444.62 | 442.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 440.70 | 443.84 | 441.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 442.55 | 443.86 | 442.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 433.50 | 441.65 | 441.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 433.50 | 441.65 | 441.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 12:15:00 | 431.95 | 435.41 | 437.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 440.90 | 435.25 | 436.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 440.90 | 435.25 | 436.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 440.90 | 435.25 | 436.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 440.90 | 435.25 | 436.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 439.20 | 436.04 | 436.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 436.50 | 435.90 | 436.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 414.67 | 423.40 | 427.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 424.55 | 423.62 | 426.66 | SL hit (close>ema200) qty=0.50 sl=423.62 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 431.20 | 424.66 | 424.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 435.15 | 428.72 | 426.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 14:15:00 | 431.75 | 432.78 | 429.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:45:00 | 432.50 | 432.78 | 429.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 430.20 | 432.14 | 430.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 430.65 | 432.14 | 430.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 432.00 | 432.11 | 430.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 430.35 | 432.11 | 430.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 429.00 | 431.49 | 430.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 429.00 | 431.49 | 430.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 430.40 | 431.27 | 430.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 431.25 | 431.27 | 430.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 426.50 | 429.35 | 429.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 426.50 | 429.35 | 429.58 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 431.10 | 429.90 | 429.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 434.45 | 431.25 | 430.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 429.85 | 432.47 | 431.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 429.85 | 432.47 | 431.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 429.85 | 432.47 | 431.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:00:00 | 429.85 | 432.47 | 431.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 434.35 | 432.84 | 431.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 433.65 | 432.84 | 431.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 432.45 | 432.76 | 431.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:45:00 | 432.40 | 432.76 | 431.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 433.00 | 432.81 | 432.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:45:00 | 432.95 | 432.81 | 432.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 434.50 | 433.15 | 432.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:45:00 | 434.00 | 433.15 | 432.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 436.20 | 433.76 | 432.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 434.60 | 433.76 | 432.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 435.80 | 437.09 | 435.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 435.80 | 437.09 | 435.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 439.30 | 437.53 | 435.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:30:00 | 434.95 | 437.53 | 435.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 439.65 | 441.69 | 439.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 439.65 | 441.69 | 439.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 438.60 | 441.07 | 439.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 438.60 | 441.07 | 439.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 438.10 | 440.48 | 439.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 437.70 | 440.48 | 439.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 436.55 | 439.69 | 439.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 436.10 | 439.69 | 439.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 436.90 | 439.13 | 438.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 436.50 | 439.13 | 438.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 433.15 | 437.94 | 438.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 10:15:00 | 431.10 | 435.21 | 436.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 418.50 | 418.44 | 423.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 418.50 | 418.44 | 423.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 419.30 | 418.91 | 422.58 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 428.45 | 424.53 | 424.41 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 407.80 | 421.79 | 423.45 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 424.00 | 420.50 | 420.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 424.70 | 421.34 | 420.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 421.75 | 422.25 | 421.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:00:00 | 421.75 | 422.25 | 421.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 425.15 | 422.83 | 421.67 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 416.85 | 421.04 | 421.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 407.00 | 417.64 | 419.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 402.35 | 399.82 | 404.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 402.35 | 399.82 | 404.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 402.35 | 399.82 | 404.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 404.95 | 399.82 | 404.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 403.00 | 400.72 | 403.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 405.40 | 400.72 | 403.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 404.70 | 401.51 | 403.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 402.70 | 402.88 | 404.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:30:00 | 402.95 | 402.73 | 403.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 402.10 | 402.73 | 403.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 417.35 | 405.78 | 405.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 417.35 | 405.78 | 405.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 421.25 | 411.85 | 408.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 414.20 | 416.57 | 412.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 414.20 | 416.57 | 412.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 414.20 | 416.57 | 412.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 414.20 | 416.57 | 412.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 411.90 | 415.64 | 412.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 411.90 | 415.64 | 412.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 414.75 | 415.46 | 412.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:30:00 | 414.90 | 415.24 | 412.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 418.75 | 414.38 | 412.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 404.50 | 418.43 | 416.94 | SL hit (close<static) qty=1.00 sl=410.65 alert=retest2 |

### Cycle 141 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 403.15 | 415.37 | 415.69 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 418.05 | 413.67 | 413.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 421.90 | 416.93 | 415.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 423.35 | 426.27 | 421.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 423.35 | 426.27 | 421.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 423.35 | 426.27 | 421.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 428.25 | 425.84 | 422.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 428.50 | 426.73 | 423.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 416.10 | 424.65 | 423.32 | SL hit (close<static) qty=1.00 sl=420.70 alert=retest2 |

### Cycle 143 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 419.75 | 422.41 | 422.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 412.00 | 418.48 | 420.44 | Break + close below crossover candle low |

### Cycle 144 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 435.15 | 421.82 | 421.78 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 419.75 | 424.54 | 424.80 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 427.55 | 424.98 | 424.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 428.90 | 426.31 | 425.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 467.25 | 468.51 | 463.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 467.25 | 468.51 | 463.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 467.25 | 468.51 | 463.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 469.50 | 468.51 | 463.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 462.50 | 464.13 | 464.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 12:15:00 | 462.50 | 464.13 | 464.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 14:15:00 | 460.50 | 462.90 | 463.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 14:15:00 | 459.90 | 459.59 | 461.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 14:15:00 | 459.90 | 459.59 | 461.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 459.90 | 459.59 | 461.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:45:00 | 460.75 | 459.59 | 461.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 466.40 | 460.86 | 461.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 466.40 | 460.86 | 461.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 464.15 | 461.52 | 461.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 461.50 | 461.52 | 461.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 463.80 | 462.11 | 461.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 463.80 | 462.11 | 461.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 470.65 | 463.81 | 462.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 465.70 | 466.87 | 464.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 465.70 | 466.87 | 464.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 462.50 | 465.78 | 464.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 469.05 | 465.78 | 464.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 470.85 | 466.80 | 465.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 473.85 | 468.80 | 466.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 457.70 | 467.97 | 467.67 | SL hit (close<static) qty=1.00 sl=460.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 460.00 | 466.38 | 466.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 452.40 | 457.17 | 461.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 459.40 | 451.44 | 454.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 459.40 | 451.44 | 454.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 459.40 | 451.44 | 454.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 464.00 | 451.44 | 454.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 467.35 | 454.62 | 455.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 467.35 | 454.62 | 455.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 465.50 | 456.80 | 456.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 469.40 | 465.50 | 462.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 465.50 | 466.27 | 463.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 465.50 | 466.27 | 463.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 456.05 | 464.28 | 463.38 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 457.35 | 461.82 | 462.35 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 467.65 | 462.86 | 462.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 469.90 | 464.93 | 463.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 12:15:00 | 464.90 | 465.96 | 464.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 12:15:00 | 464.90 | 465.96 | 464.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 464.90 | 465.96 | 464.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 465.80 | 465.96 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 464.50 | 465.67 | 464.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 465.70 | 465.67 | 464.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 467.00 | 465.93 | 464.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 469.05 | 465.93 | 464.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 453.40 | 463.93 | 464.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 453.40 | 463.93 | 464.09 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 469.55 | 463.27 | 462.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 470.70 | 464.76 | 463.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 476.10 | 478.51 | 473.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 476.10 | 478.51 | 473.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 10:15:00 | 398.00 | 2024-05-27 12:15:00 | 410.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-06-04 09:15:00 | 362.00 | 2024-06-06 11:15:00 | 383.30 | STOP_HIT | 1.00 | -5.88% |
| BUY | retest2 | 2024-06-12 09:15:00 | 390.00 | 2024-06-21 11:15:00 | 399.05 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2024-07-05 13:00:00 | 395.05 | 2024-07-12 09:15:00 | 375.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-05 13:00:00 | 395.05 | 2024-07-12 13:15:00 | 379.15 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2024-07-10 09:15:00 | 394.00 | 2024-07-15 09:15:00 | 374.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 09:15:00 | 394.00 | 2024-07-15 12:15:00 | 377.20 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2024-07-25 11:30:00 | 386.50 | 2024-08-05 09:15:00 | 425.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 09:30:00 | 386.15 | 2024-08-05 09:15:00 | 424.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-12 13:45:00 | 407.20 | 2024-08-13 14:15:00 | 408.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-12 14:15:00 | 406.65 | 2024-08-13 14:15:00 | 408.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-08-12 14:45:00 | 405.50 | 2024-08-13 14:15:00 | 408.05 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-08-13 12:00:00 | 407.05 | 2024-08-13 14:15:00 | 408.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-08-14 11:45:00 | 412.90 | 2024-08-22 14:15:00 | 422.80 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2024-08-16 10:15:00 | 413.05 | 2024-08-22 14:15:00 | 422.80 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2024-08-30 13:00:00 | 417.60 | 2024-09-03 11:15:00 | 424.35 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-09-02 09:30:00 | 417.40 | 2024-09-03 11:15:00 | 424.35 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-09-16 09:15:00 | 419.25 | 2024-09-26 12:15:00 | 431.30 | STOP_HIT | 1.00 | 2.87% |
| SELL | retest2 | 2024-10-08 12:15:00 | 410.00 | 2024-10-08 15:15:00 | 415.45 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-10-17 09:30:00 | 403.00 | 2024-10-22 09:15:00 | 382.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 403.00 | 2024-10-25 09:15:00 | 362.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-06 12:15:00 | 353.30 | 2024-11-06 13:15:00 | 356.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-18 09:30:00 | 386.25 | 2024-12-18 10:15:00 | 383.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-24 13:00:00 | 350.05 | 2025-01-01 10:15:00 | 350.60 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-12-26 10:45:00 | 350.25 | 2025-01-01 10:15:00 | 350.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-12-27 09:30:00 | 351.35 | 2025-01-01 10:15:00 | 350.60 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-12-27 14:00:00 | 350.70 | 2025-01-02 11:15:00 | 349.05 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-12-31 09:15:00 | 345.20 | 2025-01-02 11:15:00 | 349.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-31 10:00:00 | 343.25 | 2025-01-02 11:15:00 | 349.05 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-31 14:15:00 | 345.10 | 2025-01-02 11:15:00 | 349.05 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-09 13:30:00 | 336.75 | 2025-01-10 10:15:00 | 339.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-01-09 14:30:00 | 336.70 | 2025-01-10 10:15:00 | 339.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-01-10 14:00:00 | 335.65 | 2025-01-13 15:15:00 | 318.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 14:00:00 | 335.65 | 2025-01-14 09:15:00 | 327.15 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-01-23 13:00:00 | 324.05 | 2025-01-27 11:15:00 | 309.27 | PARTIAL | 0.50 | 4.56% |
| SELL | retest2 | 2025-01-24 10:00:00 | 325.55 | 2025-01-27 14:15:00 | 308.99 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-01-23 13:00:00 | 324.05 | 2025-01-27 15:15:00 | 315.60 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-01-24 10:00:00 | 325.55 | 2025-01-27 15:15:00 | 315.60 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-01-24 14:00:00 | 325.25 | 2025-01-28 09:15:00 | 307.85 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-01-24 14:00:00 | 325.25 | 2025-01-28 11:15:00 | 316.00 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2025-02-04 09:15:00 | 327.30 | 2025-02-04 10:15:00 | 322.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-02-04 14:15:00 | 324.70 | 2025-02-06 11:15:00 | 321.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-06 09:45:00 | 324.60 | 2025-02-06 11:15:00 | 321.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-02-06 10:30:00 | 324.50 | 2025-02-06 11:15:00 | 321.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-02-10 09:15:00 | 303.35 | 2025-02-11 10:15:00 | 288.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 303.35 | 2025-02-12 09:15:00 | 273.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 11:30:00 | 249.33 | 2025-03-05 13:15:00 | 252.81 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-03-17 12:45:00 | 245.25 | 2025-03-18 09:15:00 | 249.24 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-03-17 14:00:00 | 244.79 | 2025-03-18 09:15:00 | 249.24 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-03-28 11:15:00 | 257.06 | 2025-03-28 11:15:00 | 258.54 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-04-29 09:15:00 | 310.20 | 2025-05-05 11:15:00 | 305.05 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-04-29 09:45:00 | 310.00 | 2025-05-05 11:15:00 | 305.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-04-29 10:45:00 | 309.15 | 2025-05-05 11:15:00 | 305.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-05-07 13:30:00 | 303.70 | 2025-05-12 10:15:00 | 306.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-07 14:15:00 | 304.00 | 2025-05-12 10:15:00 | 306.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-08 09:15:00 | 302.95 | 2025-05-12 10:15:00 | 306.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-05-08 12:00:00 | 303.90 | 2025-05-12 10:15:00 | 306.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-19 09:15:00 | 343.10 | 2025-05-30 13:15:00 | 358.25 | STOP_HIT | 1.00 | 4.42% |
| BUY | retest2 | 2025-06-04 12:30:00 | 367.70 | 2025-06-06 12:15:00 | 366.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-06-04 13:45:00 | 367.75 | 2025-06-06 12:15:00 | 366.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-06 11:30:00 | 367.70 | 2025-06-06 12:15:00 | 366.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-06-11 12:45:00 | 365.45 | 2025-06-13 09:15:00 | 347.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 15:15:00 | 365.00 | 2025-06-13 09:15:00 | 346.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 10:45:00 | 363.80 | 2025-06-13 09:15:00 | 345.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:45:00 | 365.45 | 2025-06-13 14:15:00 | 360.95 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-06-11 15:15:00 | 365.00 | 2025-06-13 14:15:00 | 360.95 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-06-12 10:45:00 | 363.80 | 2025-06-13 14:15:00 | 360.95 | STOP_HIT | 0.50 | 0.78% |
| BUY | retest2 | 2025-06-30 12:15:00 | 385.75 | 2025-07-01 09:15:00 | 381.35 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-21 10:15:00 | 428.25 | 2025-07-25 11:15:00 | 431.00 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest1 | 2025-08-04 09:15:00 | 450.65 | 2025-08-12 09:15:00 | 463.35 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-08-14 09:15:00 | 470.70 | 2025-08-14 12:15:00 | 461.75 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-08-21 11:15:00 | 473.30 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-22 09:15:00 | 473.85 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-22 11:30:00 | 473.00 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-22 12:15:00 | 475.15 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-25 11:30:00 | 476.90 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-08-26 10:45:00 | 474.95 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-26 13:30:00 | 475.35 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-08-28 09:45:00 | 476.35 | 2025-08-28 14:15:00 | 468.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2025-09-01 09:15:00 | 464.00 | 2025-09-01 10:15:00 | 475.05 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-09-10 14:15:00 | 463.45 | 2025-09-11 10:15:00 | 467.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-15 10:45:00 | 472.75 | 2025-09-22 09:15:00 | 472.55 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-09-15 14:15:00 | 471.60 | 2025-09-22 09:15:00 | 472.55 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-09-15 15:15:00 | 471.30 | 2025-09-22 09:15:00 | 472.55 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-09-30 13:30:00 | 447.60 | 2025-09-30 15:15:00 | 453.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-01 09:15:00 | 445.00 | 2025-10-06 09:15:00 | 456.30 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-10-13 11:15:00 | 468.30 | 2025-10-15 10:15:00 | 462.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-13 13:45:00 | 472.10 | 2025-10-15 10:15:00 | 462.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-14 11:15:00 | 468.55 | 2025-10-15 10:15:00 | 462.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-14 14:30:00 | 467.95 | 2025-10-15 10:15:00 | 462.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-10-31 09:15:00 | 479.70 | 2025-10-31 11:15:00 | 472.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-05 14:30:00 | 403.15 | 2025-12-09 11:15:00 | 411.35 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-12-08 09:45:00 | 403.60 | 2025-12-09 11:15:00 | 411.35 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-12-09 10:45:00 | 404.30 | 2025-12-09 11:15:00 | 411.35 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-18 14:15:00 | 400.00 | 2025-12-19 12:15:00 | 403.25 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-26 09:15:00 | 408.85 | 2025-12-26 09:15:00 | 406.55 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-31 14:15:00 | 402.65 | 2026-01-02 14:15:00 | 404.25 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-01-01 09:15:00 | 402.65 | 2026-01-02 14:15:00 | 404.25 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-01-02 12:15:00 | 402.85 | 2026-01-02 14:15:00 | 404.25 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-01-08 13:45:00 | 416.80 | 2026-01-09 10:15:00 | 411.75 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-01-08 14:30:00 | 416.85 | 2026-01-09 10:15:00 | 411.75 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-22 11:15:00 | 380.75 | 2026-01-22 12:15:00 | 390.10 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-02-05 12:30:00 | 442.55 | 2026-02-06 09:15:00 | 433.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-02-10 12:45:00 | 436.50 | 2026-02-13 10:15:00 | 414.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:45:00 | 436.50 | 2026-02-13 12:15:00 | 424.55 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest2 | 2026-02-19 13:15:00 | 431.25 | 2026-02-20 09:15:00 | 426.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-03-17 13:45:00 | 402.70 | 2026-03-18 09:15:00 | 417.35 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2026-03-17 14:30:00 | 402.95 | 2026-03-18 09:15:00 | 417.35 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-03-17 15:00:00 | 402.10 | 2026-03-18 09:15:00 | 417.35 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-03-19 12:30:00 | 414.90 | 2026-03-23 09:15:00 | 404.50 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-03-20 09:15:00 | 418.75 | 2026-03-23 09:15:00 | 404.50 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-27 13:15:00 | 428.25 | 2026-03-30 09:15:00 | 416.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-03-27 15:00:00 | 428.50 | 2026-03-30 09:15:00 | 416.10 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-04-13 10:15:00 | 469.50 | 2026-04-15 12:15:00 | 462.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-04-17 11:15:00 | 461.50 | 2026-04-17 15:15:00 | 463.80 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-04-21 10:30:00 | 473.85 | 2026-04-22 10:15:00 | 457.70 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-05-04 15:15:00 | 469.05 | 2026-05-05 09:15:00 | 453.40 | STOP_HIT | 1.00 | -3.34% |
