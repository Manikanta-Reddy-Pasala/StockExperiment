# Elecon Engineering Co. Ltd. (ELECON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 562.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 210 |
| ALERT1 | 146 |
| ALERT2 | 146 |
| ALERT2_SKIP | 73 |
| ALERT3 | 391 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 162 |
| PARTIAL | 19 |
| TARGET_HIT | 11 |
| STOP_HIT | 154 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 184 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 125
- **Target hits / Stop hits / Partials:** 11 / 154 / 19
- **Avg / median % per leg:** 0.35% / -0.91%
- **Sum % (uncompounded):** 64.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 20 | 25.0% | 6 | 74 | 0 | 0.10% | 8.1% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| BUY @ 3rd Alert (retest2) | 79 | 20 | 25.3% | 6 | 73 | 0 | 0.14% | 11.1% |
| SELL (all) | 104 | 39 | 37.5% | 5 | 80 | 19 | 0.54% | 56.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.15% | -8.3% |
| SELL @ 3rd Alert (retest2) | 102 | 39 | 38.2% | 5 | 78 | 19 | 0.63% | 64.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.78% | -11.3% |
| retest2 (combined) | 181 | 59 | 32.6% | 11 | 151 | 19 | 0.42% | 75.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 12:15:00 | 279.77 | 285.88 | 286.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 13:15:00 | 278.08 | 284.32 | 285.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 268.10 | 266.08 | 269.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-01 09:30:00 | 268.08 | 266.08 | 269.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 269.08 | 267.12 | 269.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:45:00 | 269.95 | 267.12 | 269.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 271.35 | 267.97 | 269.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:00:00 | 271.35 | 267.97 | 269.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 271.75 | 268.72 | 269.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 268.40 | 269.37 | 269.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 14:00:00 | 270.23 | 269.50 | 269.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 09:15:00 | 276.35 | 270.69 | 270.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 276.35 | 270.69 | 270.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 13:15:00 | 277.50 | 274.38 | 272.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 274.50 | 275.05 | 273.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 10:00:00 | 274.50 | 275.05 | 273.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 274.88 | 275.01 | 273.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:45:00 | 273.73 | 275.01 | 273.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 275.40 | 278.60 | 277.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 274.55 | 278.60 | 277.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 275.83 | 278.04 | 277.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 275.77 | 278.04 | 277.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 281.33 | 281.57 | 280.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 11:45:00 | 283.45 | 282.05 | 280.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 11:15:00 | 287.00 | 291.80 | 292.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 287.00 | 291.80 | 292.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 14:15:00 | 283.45 | 288.80 | 290.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 277.10 | 275.74 | 278.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 277.10 | 275.74 | 278.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 277.10 | 275.74 | 278.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 278.73 | 275.74 | 278.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 282.70 | 277.13 | 278.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 282.70 | 277.13 | 278.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 287.55 | 279.22 | 279.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 287.55 | 279.22 | 279.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 292.52 | 281.88 | 280.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 13:15:00 | 294.30 | 284.36 | 281.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 287.75 | 288.26 | 285.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 13:30:00 | 287.65 | 288.26 | 285.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 286.38 | 287.50 | 285.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:15:00 | 284.75 | 287.50 | 285.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 285.48 | 287.10 | 285.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:30:00 | 285.43 | 287.10 | 285.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 286.50 | 286.98 | 286.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 12:15:00 | 286.90 | 286.98 | 286.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-03 12:15:00 | 315.59 | 300.46 | 293.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 13:15:00 | 366.80 | 369.09 | 369.26 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 371.88 | 369.69 | 369.49 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 10:15:00 | 365.45 | 368.84 | 369.12 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 376.30 | 369.68 | 369.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 383.18 | 375.40 | 372.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 15:15:00 | 376.50 | 377.04 | 374.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:15:00 | 376.53 | 377.04 | 374.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 375.43 | 377.01 | 375.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 12:30:00 | 374.90 | 377.01 | 375.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 376.05 | 376.82 | 375.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:15:00 | 376.50 | 376.82 | 375.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 375.75 | 376.61 | 375.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 374.65 | 376.61 | 375.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 375.88 | 376.46 | 375.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 372.35 | 376.46 | 375.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 374.68 | 376.10 | 375.58 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 13:15:00 | 369.85 | 374.54 | 375.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 14:15:00 | 366.78 | 372.98 | 374.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 381.83 | 373.88 | 374.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 381.83 | 373.88 | 374.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 381.83 | 373.88 | 374.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:45:00 | 381.83 | 373.88 | 374.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 379.70 | 375.04 | 374.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 384.50 | 378.43 | 376.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 388.50 | 388.94 | 385.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 13:45:00 | 388.93 | 388.94 | 385.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 401.93 | 405.35 | 400.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 15:00:00 | 401.93 | 405.35 | 400.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 399.85 | 404.25 | 400.59 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 09:15:00 | 396.48 | 399.29 | 399.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 388.35 | 397.10 | 398.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 394.75 | 392.71 | 395.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 394.75 | 392.71 | 395.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 394.75 | 392.71 | 395.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:45:00 | 396.28 | 392.71 | 395.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 393.23 | 392.81 | 395.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 395.83 | 392.81 | 395.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 393.28 | 392.91 | 394.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:45:00 | 394.00 | 392.91 | 394.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 394.08 | 392.70 | 394.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 13:45:00 | 393.55 | 392.70 | 394.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 396.38 | 393.43 | 394.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 396.38 | 393.43 | 394.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 395.50 | 393.85 | 394.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 09:15:00 | 396.25 | 393.85 | 394.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 396.03 | 394.28 | 394.78 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 11:15:00 | 396.58 | 395.09 | 395.08 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 393.68 | 394.99 | 395.05 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 14:15:00 | 395.75 | 395.14 | 395.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 09:15:00 | 408.83 | 397.99 | 396.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 09:15:00 | 414.08 | 414.37 | 410.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 414.08 | 414.37 | 410.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 414.08 | 414.37 | 410.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 10:15:00 | 416.33 | 414.37 | 410.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 11:45:00 | 415.45 | 414.89 | 411.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:15:00 | 418.73 | 414.70 | 412.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 14:15:00 | 409.88 | 417.77 | 415.52 | SL hit (close<static) qty=1.00 sl=410.18 alert=retest2 |

### Cycle 15 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 423.08 | 423.93 | 423.96 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 424.65 | 424.02 | 423.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 431.20 | 425.57 | 424.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 15:15:00 | 425.00 | 427.45 | 426.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 15:15:00 | 425.00 | 427.45 | 426.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 425.00 | 427.45 | 426.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 09:15:00 | 432.65 | 427.45 | 426.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 09:15:00 | 475.92 | 464.55 | 454.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 14:15:00 | 450.85 | 462.55 | 462.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 09:15:00 | 448.30 | 457.69 | 460.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 10:15:00 | 412.10 | 410.45 | 422.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-12 10:45:00 | 414.25 | 410.45 | 422.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 384.90 | 384.97 | 390.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:30:00 | 387.50 | 384.97 | 390.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 390.38 | 386.05 | 390.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:45:00 | 393.28 | 386.05 | 390.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 383.55 | 385.55 | 389.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 13:00:00 | 381.00 | 384.64 | 389.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 10:15:00 | 381.05 | 380.82 | 385.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 12:15:00 | 361.95 | 368.31 | 374.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 12:15:00 | 362.00 | 368.31 | 374.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-22 12:15:00 | 363.45 | 358.71 | 365.38 | SL hit (close>ema200) qty=0.50 sl=358.71 alert=retest2 |

### Cycle 18 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 380.50 | 367.54 | 366.88 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 366.88 | 369.82 | 370.08 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 13:15:00 | 378.60 | 371.43 | 370.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 09:15:00 | 399.43 | 387.98 | 383.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 15:15:00 | 395.28 | 395.54 | 390.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 09:15:00 | 395.78 | 395.54 | 390.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 402.73 | 396.98 | 391.31 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 10:15:00 | 387.93 | 395.65 | 395.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 13:15:00 | 381.93 | 387.94 | 390.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 387.00 | 385.91 | 388.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 387.00 | 385.91 | 388.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 387.00 | 385.91 | 388.90 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 391.98 | 390.13 | 390.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 398.93 | 391.89 | 390.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 09:15:00 | 410.85 | 415.89 | 411.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 410.85 | 415.89 | 411.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 410.85 | 415.89 | 411.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:30:00 | 410.45 | 415.89 | 411.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 409.75 | 414.67 | 411.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 10:45:00 | 409.45 | 414.67 | 411.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 410.08 | 413.75 | 411.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 12:15:00 | 409.68 | 413.75 | 411.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 10:15:00 | 408.90 | 409.73 | 409.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 11:15:00 | 405.50 | 408.88 | 409.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 14:15:00 | 406.63 | 406.02 | 407.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-17 14:45:00 | 405.80 | 406.02 | 407.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 406.90 | 406.19 | 407.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:15:00 | 412.50 | 406.19 | 407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 411.00 | 407.16 | 407.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:30:00 | 410.00 | 407.16 | 407.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 402.45 | 406.21 | 407.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 400.95 | 404.97 | 406.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:45:00 | 401.50 | 396.75 | 398.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 12:15:00 | 404.50 | 400.66 | 400.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 12:15:00 | 404.50 | 400.66 | 400.25 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 390.50 | 399.97 | 400.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 386.98 | 394.78 | 397.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 394.03 | 393.78 | 396.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 394.03 | 393.78 | 396.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 394.03 | 393.78 | 396.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:15:00 | 382.80 | 391.89 | 395.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 363.66 | 378.18 | 386.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 12:15:00 | 381.48 | 376.75 | 383.42 | SL hit (close>ema200) qty=0.50 sl=376.75 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 397.53 | 386.62 | 386.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 401.60 | 393.87 | 390.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 442.00 | 442.11 | 436.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:15:00 | 466.00 | 442.11 | 436.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 451.88 | 457.88 | 452.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-07 11:15:00 | 451.88 | 457.88 | 452.17 | SL hit (close<ema400) qty=1.00 sl=452.17 alert=retest1 |

### Cycle 27 — SELL (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 10:15:00 | 438.75 | 449.60 | 450.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 14:15:00 | 437.28 | 443.19 | 446.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 09:15:00 | 453.03 | 444.63 | 446.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 453.03 | 444.63 | 446.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 453.03 | 444.63 | 446.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 453.03 | 444.63 | 446.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 454.43 | 446.59 | 447.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:30:00 | 456.55 | 446.59 | 447.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 12:15:00 | 452.73 | 448.36 | 448.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 09:15:00 | 457.00 | 451.06 | 449.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 15:15:00 | 452.00 | 452.80 | 451.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 15:15:00 | 452.00 | 452.80 | 451.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 452.00 | 452.80 | 451.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 456.28 | 452.80 | 451.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 09:15:00 | 449.38 | 452.23 | 451.32 | SL hit (close<static) qty=1.00 sl=450.83 alert=retest2 |

### Cycle 29 — SELL (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 09:15:00 | 445.83 | 450.48 | 450.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 439.20 | 445.78 | 448.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 443.60 | 440.34 | 443.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 443.60 | 440.34 | 443.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 443.60 | 440.34 | 443.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:30:00 | 449.50 | 440.34 | 443.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 445.43 | 441.36 | 443.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:45:00 | 445.90 | 441.36 | 443.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 448.73 | 442.83 | 444.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:00:00 | 448.73 | 442.83 | 444.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 450.90 | 444.45 | 444.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:45:00 | 450.33 | 444.45 | 444.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 443.00 | 444.27 | 444.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 14:30:00 | 446.75 | 444.27 | 444.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 449.80 | 445.20 | 444.94 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 442.60 | 446.04 | 446.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 438.20 | 442.73 | 444.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 15:15:00 | 441.85 | 441.80 | 443.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 09:15:00 | 442.25 | 441.80 | 443.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 443.85 | 442.21 | 443.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:15:00 | 441.43 | 442.32 | 443.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 480.98 | 449.83 | 446.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 480.98 | 449.83 | 446.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 10:15:00 | 487.50 | 457.36 | 450.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 09:15:00 | 470.53 | 474.04 | 463.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 10:00:00 | 470.53 | 474.04 | 463.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 464.95 | 469.55 | 465.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:45:00 | 464.60 | 469.55 | 465.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 464.48 | 468.54 | 465.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 458.55 | 468.54 | 465.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 456.05 | 466.04 | 464.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:00:00 | 456.05 | 466.04 | 464.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 454.83 | 463.80 | 463.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:45:00 | 455.80 | 463.80 | 463.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 11:15:00 | 457.48 | 462.53 | 462.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 12:15:00 | 454.00 | 460.83 | 462.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 13:15:00 | 461.78 | 461.02 | 462.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 13:15:00 | 461.78 | 461.02 | 462.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 461.78 | 461.02 | 462.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:00:00 | 461.78 | 461.02 | 462.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 455.63 | 459.94 | 461.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 09:45:00 | 453.15 | 458.38 | 460.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 10:15:00 | 454.53 | 458.38 | 460.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 10:45:00 | 454.80 | 457.61 | 459.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 11:15:00 | 465.00 | 459.08 | 460.38 | SL hit (close>static) qty=1.00 sl=461.93 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 470.03 | 462.54 | 461.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 472.48 | 464.53 | 462.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 468.00 | 469.25 | 466.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 468.00 | 469.25 | 466.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 472.00 | 469.80 | 467.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 477.33 | 469.80 | 467.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 14:15:00 | 468.65 | 471.51 | 471.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 14:15:00 | 468.65 | 471.51 | 471.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 09:15:00 | 462.05 | 469.06 | 470.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 466.95 | 465.39 | 467.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-06 15:00:00 | 466.95 | 465.39 | 467.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 461.00 | 464.53 | 466.97 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 475.15 | 468.73 | 468.58 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 464.88 | 469.53 | 469.71 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 478.63 | 471.31 | 470.48 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 15:15:00 | 467.05 | 470.32 | 470.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 11:15:00 | 463.78 | 468.09 | 469.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 463.45 | 461.56 | 465.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 10:00:00 | 463.45 | 461.56 | 465.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 464.95 | 462.95 | 464.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:30:00 | 465.78 | 462.95 | 464.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 463.63 | 463.08 | 464.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:30:00 | 464.38 | 463.08 | 464.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 467.00 | 464.09 | 464.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 463.95 | 464.09 | 464.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 463.05 | 463.88 | 464.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 14:45:00 | 459.88 | 462.62 | 463.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 10:45:00 | 459.38 | 461.70 | 463.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 11:15:00 | 459.50 | 461.70 | 463.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 10:15:00 | 459.90 | 455.90 | 456.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 459.58 | 456.63 | 457.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:30:00 | 461.50 | 456.63 | 457.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 441.68 | 454.30 | 456.05 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 436.89 | 454.30 | 456.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 436.41 | 454.30 | 456.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 436.52 | 454.30 | 456.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 436.90 | 454.30 | 456.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-22 09:15:00 | 447.85 | 446.23 | 449.14 | SL hit (close>ema200) qty=0.50 sl=446.23 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 462.98 | 449.25 | 448.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 469.08 | 455.19 | 451.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 466.63 | 467.38 | 462.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 466.63 | 467.38 | 462.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 466.28 | 468.11 | 465.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:45:00 | 466.43 | 468.11 | 465.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 467.38 | 467.96 | 465.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 472.05 | 466.23 | 465.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:45:00 | 470.05 | 466.48 | 465.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 11:15:00 | 469.48 | 466.93 | 466.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 13:30:00 | 469.35 | 467.49 | 466.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 468.58 | 467.71 | 466.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 470.23 | 467.97 | 467.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 464.68 | 467.31 | 466.80 | SL hit (close<static) qty=1.00 sl=465.78 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 453.70 | 464.59 | 465.61 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 11:15:00 | 466.55 | 463.35 | 463.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 469.50 | 465.24 | 464.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 11:15:00 | 465.15 | 465.44 | 464.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 11:15:00 | 465.15 | 465.44 | 464.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 465.15 | 465.44 | 464.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:30:00 | 465.20 | 465.44 | 464.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 467.48 | 465.99 | 464.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 480.95 | 466.29 | 465.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-15 10:15:00 | 529.05 | 514.97 | 508.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 507.38 | 516.64 | 517.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 499.25 | 509.90 | 513.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 510.30 | 508.60 | 512.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 11:45:00 | 509.50 | 508.60 | 512.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 512.33 | 509.26 | 511.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 517.60 | 509.26 | 511.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 525.00 | 512.41 | 512.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 525.00 | 512.41 | 512.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 527.00 | 515.33 | 514.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 540.13 | 523.30 | 518.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 534.53 | 538.28 | 532.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 534.53 | 538.28 | 532.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 534.53 | 538.28 | 532.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 534.53 | 538.28 | 532.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 527.42 | 536.11 | 531.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 11:00:00 | 527.42 | 536.11 | 531.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 518.83 | 532.66 | 530.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 518.83 | 532.66 | 530.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 519.85 | 528.26 | 528.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 504.53 | 523.52 | 526.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 523.78 | 520.61 | 524.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 523.78 | 520.61 | 524.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 523.78 | 520.61 | 524.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:00:00 | 523.78 | 520.61 | 524.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 528.48 | 522.19 | 524.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:00:00 | 522.50 | 522.25 | 524.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 14:15:00 | 530.23 | 526.27 | 526.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 530.23 | 526.27 | 526.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 546.45 | 530.66 | 528.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 13:15:00 | 534.42 | 538.59 | 533.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-25 14:00:00 | 534.42 | 538.59 | 533.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 518.92 | 534.65 | 532.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:45:00 | 518.05 | 534.65 | 532.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 517.50 | 531.22 | 530.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 534.50 | 531.22 | 530.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 524.60 | 529.90 | 530.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 524.60 | 529.90 | 530.15 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 553.85 | 534.04 | 531.43 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 533.38 | 537.44 | 537.93 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 543.60 | 538.67 | 538.44 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 537.00 | 538.61 | 538.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 527.88 | 534.34 | 536.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 531.28 | 528.14 | 531.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 531.28 | 528.14 | 531.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 531.28 | 528.14 | 531.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:30:00 | 524.90 | 527.11 | 530.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 09:30:00 | 523.60 | 527.46 | 529.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 13:15:00 | 498.65 | 510.54 | 518.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 09:15:00 | 497.42 | 506.07 | 514.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-13 14:15:00 | 502.43 | 496.88 | 501.78 | SL hit (close>ema200) qty=0.50 sl=496.88 alert=retest2 |

### Cycle 52 — BUY (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 10:15:00 | 521.88 | 505.51 | 504.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 524.20 | 517.30 | 511.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 10:15:00 | 517.13 | 517.26 | 512.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 11:15:00 | 515.88 | 517.26 | 512.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 513.25 | 516.30 | 513.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 15:00:00 | 513.25 | 516.30 | 513.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 516.50 | 516.34 | 513.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:15:00 | 520.80 | 516.34 | 513.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 11:15:00 | 510.50 | 515.36 | 514.04 | SL hit (close<static) qty=1.00 sl=512.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 13:15:00 | 508.85 | 513.16 | 513.21 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 525.60 | 515.34 | 514.13 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 512.55 | 516.09 | 516.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 14:15:00 | 510.10 | 514.00 | 515.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 513.92 | 513.73 | 514.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 513.92 | 513.73 | 514.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 513.92 | 513.73 | 514.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 509.03 | 513.73 | 514.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 15:00:00 | 507.45 | 504.07 | 504.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 510.95 | 506.21 | 505.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 510.95 | 506.21 | 505.71 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 501.95 | 505.43 | 505.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 12:15:00 | 500.58 | 504.46 | 505.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 471.95 | 469.92 | 478.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:30:00 | 472.00 | 469.92 | 478.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 503.03 | 476.54 | 481.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:00:00 | 503.03 | 476.54 | 481.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 494.45 | 480.12 | 482.32 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 503.33 | 484.76 | 484.23 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 14:15:00 | 490.58 | 492.65 | 492.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 471.78 | 488.41 | 490.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 12:15:00 | 477.10 | 473.92 | 479.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 13:00:00 | 477.10 | 473.92 | 479.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 479.83 | 475.10 | 479.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 462.50 | 477.68 | 479.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 11:15:00 | 439.38 | 462.27 | 469.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 10:15:00 | 416.25 | 435.44 | 451.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 439.70 | 433.12 | 432.30 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 10:15:00 | 423.80 | 432.06 | 432.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 423.08 | 426.53 | 428.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 421.40 | 420.70 | 423.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 11:00:00 | 421.40 | 420.70 | 423.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 423.80 | 421.32 | 423.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:45:00 | 424.53 | 421.32 | 423.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 424.50 | 421.96 | 423.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 12:30:00 | 424.40 | 421.96 | 423.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 423.03 | 422.17 | 423.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 15:00:00 | 420.03 | 421.74 | 423.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 440.00 | 425.20 | 424.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 440.00 | 425.20 | 424.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 445.23 | 439.20 | 433.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 11:15:00 | 468.75 | 469.13 | 460.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 11:45:00 | 468.95 | 469.13 | 460.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 475.40 | 477.30 | 470.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:30:00 | 468.40 | 477.30 | 470.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 475.90 | 476.67 | 471.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:45:00 | 472.00 | 476.67 | 471.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 472.85 | 475.91 | 471.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 485.18 | 475.91 | 471.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 497.90 | 501.79 | 502.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 497.90 | 501.79 | 502.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 10:15:00 | 489.03 | 499.24 | 501.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 09:15:00 | 504.00 | 493.94 | 496.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 504.00 | 493.94 | 496.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 504.00 | 493.94 | 496.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:00:00 | 504.00 | 493.94 | 496.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 496.58 | 494.47 | 496.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 14:45:00 | 495.13 | 496.70 | 497.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 509.63 | 499.41 | 498.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 509.63 | 499.41 | 498.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 529.00 | 507.93 | 503.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 09:15:00 | 568.50 | 574.93 | 558.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 568.50 | 574.93 | 558.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 568.50 | 574.93 | 558.95 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 566.00 | 568.67 | 568.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 15:15:00 | 563.00 | 567.53 | 568.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 14:15:00 | 574.53 | 567.70 | 567.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 14:15:00 | 574.53 | 567.70 | 567.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 574.53 | 567.70 | 567.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:00:00 | 574.53 | 567.70 | 567.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 15:15:00 | 570.67 | 568.29 | 568.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 11:15:00 | 582.75 | 573.25 | 570.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 568.03 | 573.50 | 571.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 568.03 | 573.50 | 571.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 568.03 | 573.50 | 571.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 568.03 | 573.50 | 571.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 568.95 | 572.59 | 571.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 574.00 | 572.59 | 571.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 13:15:00 | 564.83 | 571.44 | 571.42 | SL hit (close<static) qty=1.00 sl=565.13 alert=retest2 |

### Cycle 67 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 568.92 | 570.93 | 571.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 557.40 | 567.06 | 569.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 10:15:00 | 528.75 | 526.03 | 535.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 11:00:00 | 528.75 | 526.03 | 535.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 534.25 | 528.45 | 534.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:15:00 | 535.05 | 528.45 | 534.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 533.95 | 529.55 | 534.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 525.00 | 530.16 | 534.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-10 09:15:00 | 472.50 | 512.22 | 521.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 536.33 | 511.73 | 509.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 541.10 | 521.81 | 514.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 572.65 | 573.42 | 568.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:30:00 | 570.05 | 573.42 | 568.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 570.00 | 572.00 | 569.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 570.00 | 572.00 | 569.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 568.90 | 571.38 | 569.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 568.90 | 571.38 | 569.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 573.50 | 571.80 | 569.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 574.95 | 572.43 | 570.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:30:00 | 574.98 | 571.92 | 570.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 15:15:00 | 573.88 | 571.14 | 570.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 09:30:00 | 576.17 | 572.35 | 571.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 572.60 | 574.07 | 572.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 572.60 | 574.07 | 572.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 573.60 | 573.97 | 572.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 582.55 | 573.97 | 572.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 570.23 | 578.49 | 577.73 | SL hit (close<static) qty=1.00 sl=571.75 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 568.50 | 576.49 | 576.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 564.70 | 572.95 | 575.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 558.05 | 550.87 | 556.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 558.05 | 550.87 | 556.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 558.05 | 550.87 | 556.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 556.42 | 550.87 | 556.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 556.00 | 551.90 | 556.39 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 569.10 | 559.20 | 558.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 575.03 | 562.37 | 560.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 554.53 | 567.05 | 564.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 554.53 | 567.05 | 564.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 554.53 | 567.05 | 564.21 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 516.55 | 556.95 | 559.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 504.38 | 546.43 | 554.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 528.50 | 514.28 | 525.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 528.50 | 514.28 | 525.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 528.50 | 514.28 | 525.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 532.48 | 514.28 | 525.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 532.10 | 517.84 | 526.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 532.50 | 517.84 | 526.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 522.05 | 518.68 | 525.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:45:00 | 520.73 | 519.62 | 525.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 536.92 | 524.81 | 526.20 | SL hit (close>static) qty=1.00 sl=533.17 alert=retest2 |

### Cycle 72 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 545.17 | 528.88 | 527.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 551.65 | 538.64 | 533.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 640.75 | 642.29 | 614.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:30:00 | 643.73 | 642.29 | 614.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 642.50 | 650.40 | 647.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:45:00 | 644.95 | 650.40 | 647.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 642.60 | 648.84 | 647.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 640.50 | 648.84 | 647.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 647.50 | 648.09 | 647.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:45:00 | 655.15 | 648.40 | 647.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 651.98 | 649.02 | 647.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 664.88 | 651.84 | 649.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 15:00:00 | 652.45 | 653.57 | 651.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 651.50 | 653.16 | 651.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 650.20 | 653.16 | 651.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 652.38 | 653.00 | 651.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 656.95 | 652.79 | 651.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:00:00 | 670.20 | 656.93 | 653.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 661.63 | 663.47 | 663.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 661.63 | 663.47 | 663.53 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 665.98 | 663.31 | 662.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 676.40 | 666.95 | 664.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 672.45 | 672.45 | 668.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 672.45 | 672.45 | 668.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 670.88 | 675.19 | 672.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 670.55 | 675.19 | 672.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 667.50 | 673.66 | 671.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 667.50 | 673.66 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 676.78 | 674.66 | 672.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 673.53 | 674.66 | 672.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 672.10 | 674.98 | 673.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 672.10 | 674.98 | 673.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 671.53 | 674.29 | 673.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 15:15:00 | 675.00 | 674.29 | 673.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 669.55 | 680.60 | 678.26 | SL hit (close<static) qty=1.00 sl=670.20 alert=retest2 |

### Cycle 75 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 640.28 | 672.54 | 674.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 637.98 | 665.63 | 671.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 652.78 | 652.54 | 660.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 11:00:00 | 652.78 | 652.54 | 660.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 640.75 | 635.68 | 643.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 646.00 | 635.68 | 643.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 639.25 | 636.39 | 643.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 641.17 | 636.39 | 643.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 640.70 | 638.41 | 642.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:30:00 | 641.50 | 638.41 | 642.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 646.28 | 639.98 | 642.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 646.28 | 639.98 | 642.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 646.00 | 641.18 | 643.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 649.40 | 641.18 | 643.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 653.13 | 645.78 | 644.93 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 642.50 | 644.30 | 644.45 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 654.28 | 646.30 | 645.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 656.80 | 649.51 | 647.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 656.00 | 658.12 | 654.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:00:00 | 656.00 | 658.12 | 654.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 650.88 | 656.67 | 653.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 655.00 | 656.67 | 653.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 656.45 | 656.63 | 654.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 595.73 | 656.63 | 654.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 604.25 | 646.15 | 649.67 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 11:15:00 | 671.70 | 645.12 | 644.74 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 09:15:00 | 640.00 | 645.01 | 645.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 612.05 | 630.90 | 637.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 14:15:00 | 631.95 | 621.70 | 629.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 14:15:00 | 631.95 | 621.70 | 629.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 631.95 | 621.70 | 629.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 631.95 | 621.70 | 629.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 627.10 | 622.78 | 628.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 624.45 | 623.99 | 628.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 10:15:00 | 624.90 | 623.99 | 628.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 626.00 | 612.22 | 616.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 628.00 | 619.09 | 618.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 628.00 | 619.09 | 618.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 12:15:00 | 640.15 | 627.33 | 623.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 627.30 | 638.37 | 634.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 15:15:00 | 627.30 | 638.37 | 634.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 627.30 | 638.37 | 634.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 630.50 | 638.37 | 634.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 630.35 | 636.76 | 633.81 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 627.55 | 632.51 | 632.81 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 634.90 | 633.19 | 633.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 14:15:00 | 636.90 | 634.38 | 633.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 632.70 | 634.43 | 633.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 632.70 | 634.43 | 633.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 632.70 | 634.43 | 633.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 636.50 | 635.55 | 634.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 12:15:00 | 628.50 | 633.59 | 633.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 628.50 | 633.59 | 633.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 626.50 | 632.17 | 633.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 585.35 | 584.57 | 600.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:15:00 | 568.60 | 585.41 | 596.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 589.00 | 581.19 | 590.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 589.00 | 581.19 | 590.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 594.95 | 583.94 | 590.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 594.95 | 583.94 | 590.54 | SL hit (close>ema400) qty=1.00 sl=590.54 alert=retest1 |

### Cycle 86 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 596.10 | 586.77 | 586.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 601.00 | 590.98 | 588.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 599.20 | 600.14 | 595.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 599.20 | 600.14 | 595.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 596.45 | 599.40 | 595.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 588.70 | 599.40 | 595.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 586.45 | 596.81 | 594.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 584.80 | 596.81 | 594.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 588.25 | 595.10 | 593.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:45:00 | 591.70 | 594.27 | 593.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 591.05 | 594.27 | 593.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 591.15 | 593.73 | 593.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 13:45:00 | 592.65 | 593.51 | 593.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 596.00 | 594.37 | 593.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 600.60 | 594.37 | 593.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 15:15:00 | 619.70 | 625.53 | 625.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 619.70 | 625.53 | 625.74 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 627.70 | 623.64 | 623.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 629.70 | 624.85 | 624.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 630.00 | 630.21 | 627.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 630.00 | 630.21 | 627.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 630.00 | 630.21 | 627.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:45:00 | 628.50 | 630.21 | 627.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 626.10 | 629.39 | 627.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:15:00 | 626.50 | 629.39 | 627.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 627.05 | 628.92 | 627.29 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 616.35 | 624.82 | 625.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 615.90 | 621.78 | 624.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 625.00 | 621.77 | 623.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 11:15:00 | 625.00 | 621.77 | 623.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 625.00 | 621.77 | 623.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 625.55 | 621.77 | 623.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 622.40 | 621.89 | 623.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 621.35 | 621.78 | 623.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 639.70 | 625.73 | 624.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 639.70 | 625.73 | 624.73 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 625.25 | 628.36 | 628.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 622.65 | 627.22 | 627.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 626.80 | 626.11 | 627.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 626.80 | 626.11 | 627.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 626.50 | 626.18 | 627.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 627.95 | 626.18 | 627.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 625.00 | 625.95 | 626.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:30:00 | 628.15 | 625.95 | 626.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 627.25 | 626.21 | 626.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 627.25 | 626.21 | 626.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 626.75 | 626.32 | 626.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 627.75 | 626.32 | 626.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 625.10 | 626.07 | 626.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 626.95 | 626.07 | 626.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 632.45 | 627.35 | 627.25 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 621.00 | 627.20 | 627.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 618.95 | 625.55 | 626.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 624.35 | 622.75 | 624.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 624.35 | 622.75 | 624.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 624.35 | 622.75 | 624.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 624.35 | 622.75 | 624.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 631.25 | 624.45 | 625.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 630.60 | 624.45 | 625.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 629.60 | 625.48 | 625.56 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 630.35 | 626.46 | 626.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 631.05 | 627.37 | 626.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 658.55 | 658.74 | 650.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 14:00:00 | 658.55 | 658.74 | 650.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 697.00 | 697.61 | 689.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 695.50 | 697.61 | 689.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 694.50 | 700.03 | 696.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 694.50 | 700.03 | 696.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 695.80 | 699.18 | 696.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 694.85 | 699.18 | 696.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 697.40 | 698.83 | 696.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 12:30:00 | 705.25 | 699.12 | 697.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 14:15:00 | 690.15 | 695.86 | 696.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 690.15 | 695.86 | 696.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 687.15 | 693.07 | 694.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 636.50 | 634.71 | 648.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 636.50 | 634.71 | 648.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 641.75 | 636.96 | 646.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 644.05 | 636.96 | 646.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 648.00 | 639.17 | 647.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 648.00 | 639.17 | 647.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 647.60 | 640.86 | 647.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:15:00 | 645.00 | 641.77 | 646.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 15:15:00 | 656.55 | 647.70 | 648.60 | SL hit (close>static) qty=1.00 sl=655.10 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 685.00 | 655.16 | 651.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 691.75 | 662.48 | 655.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 686.00 | 686.32 | 673.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:45:00 | 685.30 | 686.32 | 673.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 704.95 | 716.57 | 713.47 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 655.50 | 702.78 | 707.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 13:15:00 | 642.00 | 690.63 | 701.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 604.10 | 597.03 | 617.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:15:00 | 603.80 | 597.03 | 617.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 566.00 | 555.83 | 564.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 565.00 | 555.83 | 564.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 558.15 | 556.30 | 563.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 556.80 | 555.78 | 562.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:00:00 | 553.70 | 555.78 | 562.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:00:00 | 556.95 | 552.04 | 553.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 11:15:00 | 563.20 | 555.96 | 555.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 563.20 | 555.96 | 555.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 565.45 | 557.86 | 556.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 581.50 | 582.13 | 574.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 09:30:00 | 578.35 | 582.13 | 574.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 579.00 | 581.16 | 577.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 585.05 | 581.16 | 577.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:45:00 | 583.95 | 582.15 | 578.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:00:00 | 583.95 | 591.96 | 590.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 577.30 | 587.14 | 588.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 577.30 | 587.14 | 588.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 574.05 | 583.15 | 586.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 575.35 | 573.46 | 578.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 15:00:00 | 575.35 | 573.46 | 578.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 573.70 | 570.89 | 575.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 573.70 | 570.89 | 575.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 555.00 | 548.23 | 556.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 537.30 | 548.23 | 556.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:45:00 | 540.00 | 550.76 | 553.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 538.00 | 548.81 | 552.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 573.00 | 554.45 | 553.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 573.00 | 554.45 | 553.08 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 555.25 | 559.08 | 559.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 13:15:00 | 553.05 | 557.26 | 558.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 561.00 | 558.01 | 558.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 561.00 | 558.01 | 558.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 561.00 | 558.01 | 558.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 561.00 | 558.01 | 558.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 562.00 | 558.81 | 559.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 576.85 | 558.81 | 559.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 578.85 | 562.82 | 560.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 595.95 | 575.52 | 568.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 571.25 | 578.55 | 571.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 571.25 | 578.55 | 571.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 571.25 | 578.55 | 571.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:30:00 | 578.70 | 575.23 | 572.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 12:15:00 | 574.25 | 579.83 | 579.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 574.25 | 579.83 | 579.94 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 583.65 | 580.59 | 580.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 585.75 | 581.63 | 580.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 633.75 | 634.32 | 621.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 633.75 | 634.32 | 621.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 626.00 | 630.87 | 628.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 626.00 | 630.87 | 628.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 630.95 | 630.89 | 628.62 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 15:15:00 | 627.00 | 627.54 | 627.60 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 631.30 | 628.29 | 627.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 642.90 | 631.21 | 629.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 642.70 | 653.93 | 647.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 642.70 | 653.93 | 647.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 642.70 | 653.93 | 647.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 641.50 | 653.93 | 647.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 644.70 | 652.08 | 647.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 646.60 | 652.08 | 647.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 651.55 | 651.98 | 648.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:45:00 | 645.25 | 651.98 | 648.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 650.50 | 651.67 | 649.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 653.35 | 651.67 | 649.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 647.85 | 650.91 | 649.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:15:00 | 646.45 | 650.91 | 649.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 649.45 | 650.62 | 649.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 645.80 | 650.62 | 649.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 649.55 | 650.40 | 649.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 647.30 | 650.40 | 649.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 649.10 | 650.14 | 649.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 648.05 | 650.14 | 649.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 648.45 | 649.80 | 649.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 648.60 | 649.80 | 649.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 649.35 | 649.71 | 649.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:15:00 | 649.00 | 649.71 | 649.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 649.00 | 649.57 | 649.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 650.55 | 649.57 | 649.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 646.50 | 648.51 | 648.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 646.50 | 648.51 | 648.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 13:15:00 | 645.30 | 647.27 | 648.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 649.55 | 647.34 | 647.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 649.55 | 647.34 | 647.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 649.55 | 647.34 | 647.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 638.30 | 645.07 | 646.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 606.38 | 618.99 | 628.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 617.55 | 617.51 | 623.84 | SL hit (close>ema200) qty=0.50 sl=617.51 alert=retest2 |

### Cycle 108 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 617.50 | 611.74 | 611.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 626.25 | 614.64 | 612.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 629.45 | 632.74 | 625.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 629.45 | 632.74 | 625.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 629.45 | 632.74 | 625.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 629.45 | 632.74 | 625.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 627.85 | 634.86 | 630.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 627.85 | 634.86 | 630.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 626.40 | 633.17 | 630.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 626.40 | 633.17 | 630.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 624.85 | 631.50 | 629.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 624.85 | 631.50 | 629.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 626.90 | 628.53 | 628.70 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 634.00 | 629.63 | 629.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 09:15:00 | 641.50 | 632.00 | 630.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 14:15:00 | 635.65 | 636.61 | 633.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 15:00:00 | 635.65 | 636.61 | 633.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 633.00 | 635.89 | 633.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 635.50 | 635.89 | 633.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 634.05 | 635.52 | 633.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 632.55 | 635.52 | 633.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 639.70 | 636.36 | 634.25 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 631.20 | 634.02 | 634.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 626.80 | 631.25 | 632.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 637.15 | 632.43 | 633.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 637.15 | 632.43 | 633.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 637.15 | 632.43 | 633.24 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 640.05 | 633.96 | 633.86 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 633.25 | 633.80 | 633.81 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 15:15:00 | 635.20 | 634.08 | 633.93 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 625.10 | 632.28 | 633.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 609.85 | 627.80 | 631.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 611.00 | 610.40 | 616.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:15:00 | 610.40 | 610.40 | 616.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 609.90 | 610.22 | 615.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 601.55 | 608.15 | 613.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:45:00 | 597.15 | 605.89 | 611.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 598.10 | 606.72 | 608.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 571.47 | 586.49 | 595.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 567.29 | 586.49 | 595.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 568.20 | 586.49 | 595.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-14 09:15:00 | 541.39 | 561.99 | 578.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 587.05 | 572.35 | 572.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 617.90 | 583.61 | 577.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 593.25 | 593.26 | 584.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 12:00:00 | 593.25 | 593.26 | 584.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 591.05 | 594.02 | 589.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 591.10 | 594.02 | 589.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 589.70 | 591.89 | 589.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 594.40 | 591.89 | 589.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 595.00 | 592.51 | 590.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 601.50 | 594.22 | 591.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 586.15 | 598.98 | 599.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 586.15 | 598.98 | 599.05 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 11:15:00 | 604.15 | 600.01 | 599.51 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 577.50 | 595.51 | 597.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 544.00 | 585.21 | 592.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 580.60 | 574.92 | 585.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 580.60 | 574.92 | 585.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 563.00 | 572.54 | 583.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 582.80 | 572.54 | 583.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 520.15 | 521.34 | 531.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 533.45 | 521.34 | 531.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 534.60 | 522.98 | 530.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 534.60 | 522.98 | 530.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 535.90 | 525.56 | 530.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 14:30:00 | 528.70 | 526.94 | 530.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 11:15:00 | 536.30 | 527.33 | 527.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 536.30 | 527.33 | 527.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 538.50 | 531.99 | 529.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 530.60 | 532.82 | 530.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 530.60 | 532.82 | 530.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 530.60 | 532.82 | 530.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 529.50 | 532.82 | 530.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 529.80 | 532.22 | 530.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 530.00 | 532.22 | 530.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 535.80 | 532.93 | 530.89 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 494.35 | 524.50 | 527.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 487.10 | 513.32 | 521.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 493.95 | 490.89 | 502.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:00:00 | 493.95 | 490.89 | 502.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 493.10 | 492.76 | 499.79 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 513.30 | 501.64 | 501.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 520.85 | 512.64 | 507.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 12:15:00 | 514.15 | 514.38 | 510.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 13:15:00 | 509.40 | 513.39 | 510.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 509.40 | 513.39 | 510.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 509.40 | 513.39 | 510.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 511.50 | 513.01 | 510.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:15:00 | 508.00 | 513.01 | 510.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 508.00 | 512.01 | 509.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 514.00 | 512.01 | 509.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 509.40 | 511.49 | 509.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:15:00 | 509.60 | 511.49 | 509.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 509.35 | 511.06 | 509.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:15:00 | 511.85 | 511.06 | 509.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 13:15:00 | 503.85 | 509.63 | 509.51 | SL hit (close<static) qty=1.00 sl=506.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 14:15:00 | 506.85 | 509.07 | 509.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 481.55 | 503.05 | 506.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 468.60 | 468.33 | 481.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 14:00:00 | 461.50 | 467.52 | 478.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 478.45 | 469.05 | 476.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 478.45 | 469.05 | 476.45 | SL hit (close>ema400) qty=1.00 sl=476.45 alert=retest1 |

### Cycle 124 — BUY (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 15:15:00 | 484.35 | 479.58 | 479.28 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 472.15 | 478.09 | 478.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 462.85 | 475.04 | 477.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 445.15 | 441.86 | 449.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 15:00:00 | 445.15 | 441.86 | 449.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 467.05 | 447.14 | 450.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 467.05 | 447.14 | 450.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 467.35 | 451.18 | 452.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 467.45 | 451.18 | 452.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 468.80 | 454.71 | 453.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 12:15:00 | 476.10 | 467.74 | 462.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 469.80 | 471.90 | 466.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 469.80 | 471.90 | 466.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 471.75 | 471.31 | 466.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 475.30 | 472.21 | 468.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 465.15 | 468.63 | 468.12 | SL hit (close<static) qty=1.00 sl=466.45 alert=retest2 |

### Cycle 127 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 459.25 | 466.13 | 467.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 452.55 | 463.41 | 465.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 14:15:00 | 392.50 | 391.57 | 402.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:45:00 | 394.45 | 391.57 | 402.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 399.90 | 393.24 | 401.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 401.15 | 393.24 | 401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 409.70 | 396.53 | 402.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:45:00 | 412.55 | 396.53 | 402.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 408.20 | 398.86 | 402.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:30:00 | 409.15 | 398.86 | 402.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 418.20 | 405.43 | 405.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 426.80 | 413.64 | 409.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 429.40 | 433.51 | 426.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 429.40 | 433.51 | 426.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 429.40 | 433.51 | 426.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 429.40 | 433.51 | 426.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 427.80 | 432.37 | 427.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 427.80 | 432.37 | 427.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 423.45 | 430.59 | 426.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 423.45 | 430.59 | 426.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 421.40 | 428.75 | 426.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 421.40 | 428.75 | 426.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 415.80 | 424.67 | 424.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 400.60 | 418.55 | 421.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 403.25 | 394.75 | 397.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 403.25 | 394.75 | 397.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 403.25 | 394.75 | 397.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 404.20 | 394.75 | 397.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 406.20 | 397.04 | 397.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 404.90 | 397.04 | 397.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 409.80 | 399.59 | 399.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 413.60 | 402.40 | 400.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 423.85 | 424.45 | 418.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:00:00 | 423.85 | 424.45 | 418.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 443.40 | 450.41 | 443.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 443.40 | 450.41 | 443.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 440.75 | 448.47 | 442.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 438.55 | 448.47 | 442.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 438.60 | 446.50 | 442.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 438.25 | 446.50 | 442.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 440.30 | 441.70 | 441.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 440.30 | 441.70 | 441.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 443.65 | 442.09 | 441.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:30:00 | 440.65 | 442.09 | 441.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 451.00 | 443.87 | 442.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 15:00:00 | 457.00 | 447.65 | 444.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 457.40 | 449.41 | 445.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 454.55 | 451.11 | 450.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 445.50 | 449.99 | 450.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 445.50 | 449.99 | 450.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 444.15 | 447.68 | 449.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 448.30 | 446.19 | 447.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 448.30 | 446.19 | 447.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 448.30 | 446.19 | 447.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 448.30 | 446.19 | 447.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 447.85 | 446.53 | 447.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 448.10 | 446.53 | 447.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 447.85 | 446.79 | 447.77 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 451.80 | 448.43 | 448.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 453.70 | 449.92 | 449.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 438.65 | 449.70 | 449.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 438.65 | 449.70 | 449.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 438.65 | 449.70 | 449.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 438.65 | 449.70 | 449.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 442.50 | 448.26 | 449.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 433.70 | 442.42 | 445.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 421.10 | 418.13 | 427.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 421.10 | 418.13 | 427.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 421.10 | 418.13 | 427.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 417.90 | 418.54 | 426.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:45:00 | 419.00 | 418.74 | 425.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:30:00 | 418.50 | 419.76 | 425.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 418.00 | 419.41 | 424.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 422.30 | 415.70 | 418.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 434.40 | 421.91 | 421.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 434.40 | 421.91 | 421.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 450.05 | 433.94 | 427.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 503.25 | 505.94 | 493.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 503.25 | 505.94 | 493.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 520.80 | 529.08 | 519.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 518.00 | 529.08 | 519.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 514.75 | 526.21 | 518.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 514.75 | 526.21 | 518.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 513.85 | 523.74 | 518.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 508.00 | 523.74 | 518.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 517.70 | 522.05 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 517.70 | 522.05 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 518.00 | 521.24 | 518.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:30:00 | 518.45 | 521.24 | 518.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 516.75 | 520.34 | 518.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 521.35 | 520.34 | 518.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 536.75 | 523.63 | 519.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:15:00 | 542.65 | 523.63 | 519.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-06 09:15:00 | 596.92 | 577.25 | 569.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 662.65 | 669.13 | 669.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 659.65 | 666.08 | 667.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 669.95 | 664.00 | 666.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 669.95 | 664.00 | 666.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 669.95 | 664.00 | 666.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 669.95 | 664.00 | 666.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 675.50 | 666.30 | 666.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 669.55 | 666.30 | 666.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 673.60 | 667.76 | 667.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 681.60 | 671.93 | 669.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 677.80 | 678.36 | 674.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:45:00 | 676.95 | 678.36 | 674.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 676.45 | 677.76 | 674.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 676.45 | 677.76 | 674.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 679.90 | 678.19 | 675.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 694.95 | 678.19 | 675.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 673.75 | 679.74 | 679.05 | SL hit (close<static) qty=1.00 sl=675.30 alert=retest2 |

### Cycle 137 — SELL (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 12:15:00 | 673.70 | 678.53 | 678.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 672.80 | 677.38 | 678.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 15:15:00 | 678.65 | 676.46 | 677.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 15:15:00 | 678.65 | 676.46 | 677.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 678.65 | 676.46 | 677.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 674.65 | 676.46 | 677.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 680.45 | 677.25 | 677.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 680.35 | 677.25 | 677.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 684.50 | 678.70 | 678.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 688.80 | 681.47 | 679.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 680.05 | 685.96 | 683.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 13:15:00 | 680.05 | 685.96 | 683.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 680.05 | 685.96 | 683.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 680.05 | 685.96 | 683.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 678.00 | 684.37 | 683.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 670.00 | 684.37 | 683.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 682.00 | 683.90 | 682.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 684.05 | 683.90 | 682.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 682.15 | 683.39 | 682.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 682.35 | 682.73 | 682.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 681.15 | 682.42 | 682.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 681.15 | 682.42 | 682.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 679.00 | 681.73 | 682.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 684.70 | 681.56 | 681.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 684.70 | 681.56 | 681.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 684.70 | 681.56 | 681.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 684.70 | 681.56 | 681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 688.45 | 682.94 | 682.52 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 671.10 | 680.57 | 681.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 664.70 | 675.45 | 678.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 684.50 | 675.27 | 678.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 684.50 | 675.27 | 678.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 684.50 | 675.27 | 678.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 684.50 | 675.27 | 678.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 686.05 | 677.42 | 678.80 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 691.65 | 680.27 | 679.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 693.75 | 682.96 | 681.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 693.70 | 696.91 | 690.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 12:45:00 | 696.40 | 696.91 | 690.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 689.95 | 695.18 | 691.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 689.95 | 695.18 | 691.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 691.00 | 694.35 | 691.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 693.65 | 694.35 | 691.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 690.70 | 693.62 | 690.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 699.00 | 692.89 | 691.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 696.35 | 699.03 | 696.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 688.65 | 694.31 | 694.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 688.65 | 694.31 | 694.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 684.70 | 692.39 | 693.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 694.50 | 691.21 | 692.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 694.50 | 691.21 | 692.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 694.50 | 691.21 | 692.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 694.50 | 691.21 | 692.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 692.70 | 691.51 | 692.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 699.00 | 691.51 | 692.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 694.65 | 692.13 | 692.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 691.40 | 692.13 | 692.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:30:00 | 690.55 | 689.21 | 691.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 699.70 | 692.08 | 691.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 699.70 | 692.08 | 691.74 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 688.65 | 691.80 | 692.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 684.60 | 689.60 | 690.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 665.90 | 665.84 | 671.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 666.45 | 665.84 | 671.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 669.70 | 666.57 | 670.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 669.70 | 666.57 | 670.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 667.65 | 666.79 | 670.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 663.30 | 666.79 | 670.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 664.65 | 666.60 | 669.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 630.13 | 642.88 | 651.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 631.42 | 642.88 | 651.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 640.50 | 637.31 | 644.24 | SL hit (close>ema200) qty=0.50 sl=637.31 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 642.10 | 636.10 | 635.55 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 631.80 | 636.15 | 636.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 13:15:00 | 628.20 | 633.94 | 635.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 646.90 | 634.40 | 634.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 646.90 | 634.40 | 634.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 646.90 | 634.40 | 634.86 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 652.00 | 637.92 | 636.41 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 639.65 | 644.82 | 645.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 633.85 | 641.59 | 643.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 643.05 | 640.41 | 642.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 643.05 | 640.41 | 642.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 643.05 | 640.41 | 642.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 643.05 | 640.41 | 642.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 641.75 | 640.68 | 642.18 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 651.20 | 643.62 | 643.31 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 642.95 | 643.89 | 644.00 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 649.15 | 644.94 | 644.46 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 641.05 | 644.48 | 644.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 635.60 | 641.97 | 643.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 641.50 | 639.15 | 641.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 641.50 | 639.15 | 641.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 641.50 | 639.15 | 641.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 641.50 | 639.15 | 641.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 639.55 | 639.23 | 641.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 638.50 | 639.23 | 641.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 638.60 | 639.10 | 640.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:15:00 | 642.65 | 639.10 | 640.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 647.05 | 640.69 | 641.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 648.05 | 640.69 | 641.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 648.30 | 642.21 | 642.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 651.30 | 644.03 | 642.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 651.30 | 652.61 | 648.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 651.30 | 652.61 | 648.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 651.30 | 652.61 | 648.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:15:00 | 645.60 | 652.61 | 648.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 646.00 | 651.29 | 648.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 644.40 | 651.29 | 648.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 647.00 | 649.37 | 648.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 649.25 | 649.37 | 648.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 649.95 | 649.49 | 648.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:15:00 | 643.80 | 649.49 | 648.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 644.80 | 648.55 | 648.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 639.00 | 648.55 | 648.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 646.80 | 648.20 | 647.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 646.80 | 648.20 | 647.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 653.00 | 649.16 | 648.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:15:00 | 642.70 | 649.16 | 648.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 617.30 | 642.79 | 645.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 613.90 | 623.57 | 628.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 615.35 | 613.41 | 619.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-17 11:00:00 | 615.35 | 613.41 | 619.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 600.65 | 600.27 | 603.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 600.00 | 600.27 | 603.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 599.00 | 598.03 | 600.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 570.00 | 579.66 | 585.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 569.05 | 579.66 | 585.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 569.75 | 569.66 | 576.33 | SL hit (close>ema200) qty=0.50 sl=569.66 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 581.50 | 577.61 | 577.24 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 570.90 | 577.52 | 577.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 562.95 | 569.51 | 572.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 568.40 | 565.89 | 569.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 568.40 | 565.89 | 569.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 568.40 | 565.89 | 569.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 568.40 | 565.89 | 569.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 572.40 | 567.19 | 569.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 572.35 | 567.19 | 569.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 572.05 | 568.16 | 569.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 572.05 | 568.16 | 569.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 569.00 | 569.40 | 570.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:15:00 | 573.35 | 569.40 | 570.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 572.00 | 569.92 | 570.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 568.40 | 569.73 | 570.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 14:15:00 | 574.15 | 570.74 | 570.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 574.15 | 570.74 | 570.49 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 564.60 | 569.87 | 570.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 561.35 | 568.17 | 569.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 542.70 | 541.72 | 546.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 542.70 | 541.72 | 546.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 542.70 | 541.72 | 546.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 539.00 | 541.18 | 545.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 539.45 | 541.41 | 544.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 560.80 | 546.76 | 546.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 560.80 | 546.76 | 546.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 568.45 | 551.10 | 548.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 556.30 | 561.34 | 555.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 556.30 | 561.34 | 555.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 556.30 | 561.34 | 555.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 556.30 | 561.34 | 555.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 554.25 | 559.92 | 555.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 554.75 | 559.92 | 555.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 554.00 | 558.74 | 555.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 553.00 | 558.74 | 555.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 552.00 | 555.61 | 554.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 552.10 | 555.61 | 554.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 553.30 | 554.66 | 554.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 553.30 | 554.66 | 554.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 556.65 | 555.06 | 554.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 557.05 | 555.06 | 554.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 551.60 | 554.37 | 554.43 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 556.70 | 554.78 | 554.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 562.60 | 556.34 | 555.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 10:15:00 | 575.60 | 575.69 | 570.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 577.10 | 575.69 | 570.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 570.55 | 574.66 | 570.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 570.55 | 574.66 | 570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 570.70 | 573.87 | 570.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 570.00 | 573.87 | 570.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 566.45 | 572.39 | 570.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 566.45 | 572.39 | 570.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 564.55 | 570.82 | 569.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 564.55 | 570.82 | 569.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 563.55 | 568.35 | 568.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 558.45 | 563.15 | 565.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 560.80 | 559.97 | 563.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 560.80 | 559.97 | 563.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 560.80 | 559.97 | 563.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 561.95 | 559.97 | 563.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 567.00 | 561.37 | 563.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 567.00 | 561.37 | 563.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 569.00 | 562.90 | 564.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 569.00 | 562.90 | 564.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 574.00 | 565.12 | 565.01 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 562.00 | 564.94 | 565.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 555.05 | 562.51 | 564.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 548.60 | 545.34 | 551.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 548.60 | 545.34 | 551.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 551.50 | 546.57 | 551.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 551.30 | 546.57 | 551.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 551.10 | 547.48 | 551.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 547.20 | 548.46 | 551.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 558.60 | 551.59 | 552.35 | SL hit (close>static) qty=1.00 sl=552.65 alert=retest2 |

### Cycle 166 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 556.05 | 553.37 | 553.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 558.50 | 554.47 | 553.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 558.05 | 558.53 | 556.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 558.05 | 558.53 | 556.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 571.55 | 561.39 | 558.12 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 555.80 | 558.83 | 559.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 552.00 | 556.87 | 558.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 556.20 | 554.03 | 555.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 556.20 | 554.03 | 555.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 556.20 | 554.03 | 555.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 556.20 | 554.03 | 555.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 552.15 | 553.65 | 555.46 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 561.00 | 556.49 | 556.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 562.50 | 557.69 | 556.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 561.40 | 563.95 | 561.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 561.40 | 563.95 | 561.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 561.40 | 563.95 | 561.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 561.05 | 563.95 | 561.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 561.00 | 563.36 | 561.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 567.55 | 563.36 | 561.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-16 10:15:00 | 624.30 | 606.61 | 599.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 614.50 | 620.42 | 621.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 611.30 | 618.60 | 620.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 563.60 | 563.30 | 571.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 563.60 | 563.30 | 571.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 563.60 | 563.30 | 571.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 563.60 | 563.30 | 571.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 567.40 | 563.55 | 568.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 567.40 | 563.55 | 568.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 569.00 | 564.64 | 568.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 577.30 | 564.64 | 568.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 587.70 | 569.25 | 570.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 582.60 | 569.25 | 570.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 584.20 | 572.24 | 571.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 593.50 | 587.75 | 583.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 610.15 | 613.11 | 603.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 610.15 | 613.11 | 603.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 606.70 | 608.88 | 605.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 614.65 | 608.88 | 605.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 602.10 | 606.72 | 605.03 | SL hit (close<static) qty=1.00 sl=604.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 596.30 | 603.69 | 604.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 591.60 | 601.27 | 603.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 539.15 | 538.75 | 550.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:45:00 | 539.30 | 538.75 | 550.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 547.95 | 541.69 | 546.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 551.40 | 541.69 | 546.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 546.50 | 542.66 | 546.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:00:00 | 543.80 | 542.88 | 546.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 544.00 | 543.54 | 546.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:00:00 | 543.25 | 542.11 | 543.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 543.75 | 542.30 | 543.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 545.05 | 542.85 | 543.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 545.05 | 542.85 | 543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 545.50 | 543.38 | 543.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 545.95 | 543.38 | 543.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 544.95 | 543.73 | 543.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 544.95 | 543.73 | 543.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 553.50 | 545.68 | 544.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 559.50 | 561.73 | 556.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:45:00 | 559.10 | 561.73 | 556.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 562.45 | 564.00 | 560.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 561.70 | 564.00 | 560.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 560.05 | 563.21 | 560.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 560.05 | 563.21 | 560.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 560.30 | 562.63 | 560.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 560.30 | 562.63 | 560.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 556.75 | 561.45 | 560.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 556.75 | 561.45 | 560.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 557.80 | 560.72 | 560.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 563.40 | 560.72 | 560.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 555.90 | 560.01 | 560.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 555.90 | 560.01 | 560.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 551.90 | 556.68 | 558.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 561.20 | 557.58 | 558.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 561.20 | 557.58 | 558.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 561.20 | 557.58 | 558.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 561.20 | 557.58 | 558.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 557.10 | 557.49 | 558.43 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 564.45 | 559.77 | 559.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 567.30 | 561.28 | 559.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 563.85 | 565.64 | 563.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 12:15:00 | 563.85 | 565.64 | 563.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 563.85 | 565.64 | 563.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 563.85 | 565.64 | 563.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 565.65 | 565.65 | 563.78 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 556.85 | 561.81 | 562.38 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 563.50 | 561.65 | 561.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 565.45 | 562.41 | 561.95 | Break + close above crossover candle high |

### Cycle 177 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 553.45 | 561.66 | 561.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 550.70 | 559.47 | 560.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 520.65 | 519.51 | 526.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 11:15:00 | 522.35 | 519.51 | 526.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 525.60 | 521.65 | 525.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:45:00 | 525.40 | 521.65 | 525.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 526.50 | 522.62 | 525.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 529.00 | 522.62 | 525.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 534.40 | 524.97 | 526.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 534.40 | 524.97 | 526.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 534.50 | 526.88 | 526.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 538.50 | 532.09 | 529.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 533.85 | 537.35 | 534.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 13:15:00 | 533.85 | 537.35 | 534.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 533.85 | 537.35 | 534.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:00:00 | 533.85 | 537.35 | 534.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 535.65 | 537.01 | 534.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:15:00 | 533.75 | 537.01 | 534.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 533.75 | 536.36 | 534.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 543.30 | 536.36 | 534.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 540.30 | 535.87 | 534.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 538.20 | 539.15 | 537.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 538.30 | 538.01 | 537.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 540.15 | 538.44 | 537.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 542.90 | 539.32 | 538.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:30:00 | 542.70 | 540.25 | 539.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 533.20 | 537.85 | 538.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 533.20 | 537.85 | 538.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 531.40 | 535.97 | 537.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 497.10 | 495.20 | 504.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:30:00 | 497.45 | 495.20 | 504.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 501.05 | 497.13 | 503.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 500.30 | 497.13 | 503.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 501.50 | 498.00 | 503.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 502.35 | 498.00 | 503.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 503.55 | 499.55 | 503.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:45:00 | 503.85 | 499.55 | 503.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 505.05 | 500.65 | 503.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 505.05 | 500.65 | 503.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 501.95 | 500.91 | 503.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 498.85 | 501.50 | 503.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 500.10 | 501.22 | 502.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 510.80 | 502.11 | 502.18 | SL hit (close>static) qty=1.00 sl=505.40 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 509.35 | 503.56 | 502.83 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 497.40 | 502.38 | 502.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 494.10 | 500.73 | 501.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 483.35 | 481.48 | 486.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 483.35 | 481.48 | 486.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 484.05 | 481.99 | 486.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 487.00 | 483.00 | 486.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 487.10 | 483.82 | 486.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 489.35 | 483.82 | 486.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 485.60 | 484.17 | 486.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 483.40 | 484.14 | 486.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 484.95 | 476.65 | 475.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 484.95 | 476.65 | 475.79 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 472.50 | 476.03 | 476.06 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 478.20 | 476.47 | 476.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 478.80 | 477.12 | 476.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 473.85 | 476.47 | 476.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 473.85 | 476.47 | 476.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 473.85 | 476.47 | 476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 473.85 | 476.47 | 476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 477.05 | 476.58 | 476.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 478.40 | 476.58 | 476.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 479.00 | 477.07 | 476.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 481.60 | 476.85 | 476.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 478.35 | 481.40 | 481.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 478.35 | 481.40 | 481.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 475.30 | 478.47 | 480.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 474.10 | 472.85 | 475.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 474.10 | 472.85 | 475.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 474.10 | 472.85 | 475.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 475.25 | 472.85 | 475.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 477.90 | 473.86 | 475.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 481.80 | 473.86 | 475.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 477.70 | 474.63 | 475.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 475.90 | 474.88 | 475.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 480.05 | 475.91 | 476.04 | SL hit (close>static) qty=1.00 sl=478.65 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 482.80 | 477.29 | 476.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 491.35 | 481.18 | 478.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 489.55 | 491.12 | 488.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 489.55 | 491.12 | 488.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 489.55 | 491.12 | 488.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 489.20 | 491.12 | 488.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 484.55 | 489.27 | 487.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 484.55 | 489.27 | 487.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 483.20 | 488.06 | 487.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 483.20 | 488.06 | 487.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 481.55 | 486.76 | 487.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 480.00 | 485.41 | 486.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 486.15 | 483.16 | 484.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 486.15 | 483.16 | 484.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 486.15 | 483.16 | 484.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 486.15 | 483.16 | 484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 477.95 | 482.12 | 483.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 476.30 | 482.12 | 483.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 477.25 | 481.44 | 483.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 477.40 | 480.06 | 482.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:30:00 | 475.80 | 479.18 | 481.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 478.30 | 473.77 | 476.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 478.30 | 473.77 | 476.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 478.15 | 474.65 | 476.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 479.20 | 474.65 | 476.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 481.60 | 477.66 | 477.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 481.60 | 477.66 | 477.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 483.60 | 479.48 | 478.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 494.65 | 497.74 | 492.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:00:00 | 494.65 | 497.74 | 492.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 488.20 | 494.95 | 491.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 488.20 | 494.95 | 491.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 498.60 | 495.68 | 492.32 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 479.05 | 489.56 | 490.60 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 515.90 | 494.28 | 491.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 11:15:00 | 519.25 | 499.27 | 494.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 506.00 | 509.71 | 502.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 506.00 | 509.71 | 502.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 505.35 | 508.84 | 503.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 505.35 | 508.84 | 503.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 505.25 | 508.12 | 503.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 503.35 | 508.12 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 512.15 | 508.93 | 504.06 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 438.40 | 492.98 | 497.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 424.45 | 470.33 | 486.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 375.25 | 374.95 | 380.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 375.25 | 374.95 | 380.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 379.00 | 376.39 | 380.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 383.90 | 376.39 | 380.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 379.10 | 376.94 | 380.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 376.50 | 376.94 | 380.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 376.85 | 376.43 | 377.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 382.80 | 374.68 | 374.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 382.80 | 374.68 | 374.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 387.00 | 377.14 | 375.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 12:15:00 | 386.20 | 386.22 | 381.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:45:00 | 385.60 | 386.22 | 381.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 400.85 | 406.51 | 400.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 397.95 | 406.51 | 400.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 400.00 | 405.20 | 400.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 400.00 | 405.20 | 400.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 396.00 | 403.36 | 399.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 396.00 | 403.36 | 399.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 396.15 | 401.92 | 399.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 397.25 | 401.92 | 399.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 392.75 | 400.12 | 399.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 392.75 | 400.12 | 399.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 393.45 | 398.79 | 398.55 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 393.60 | 397.75 | 398.10 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 402.35 | 398.65 | 398.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 416.55 | 403.00 | 400.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 444.30 | 447.94 | 435.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 442.20 | 447.94 | 435.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 435.70 | 442.10 | 436.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 435.70 | 442.10 | 436.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 438.15 | 441.31 | 436.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 444.00 | 440.74 | 436.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 11:15:00 | 488.40 | 457.47 | 447.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 454.15 | 462.61 | 462.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 449.45 | 457.91 | 460.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 436.90 | 433.73 | 438.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 445.55 | 433.73 | 438.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 444.65 | 435.91 | 439.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 430.55 | 436.61 | 438.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 409.02 | 417.23 | 419.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 398.90 | 394.42 | 399.21 | SL hit (close>ema200) qty=0.50 sl=394.42 alert=retest2 |

### Cycle 196 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 412.25 | 403.53 | 402.52 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 401.75 | 404.11 | 404.14 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 405.90 | 404.46 | 404.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 15:15:00 | 407.25 | 405.02 | 404.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 405.05 | 405.52 | 404.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 11:15:00 | 405.05 | 405.52 | 404.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 405.05 | 405.52 | 404.96 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 14:15:00 | 402.05 | 404.39 | 404.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 399.75 | 403.19 | 403.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 375.70 | 375.38 | 382.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 378.20 | 375.38 | 382.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 382.70 | 376.80 | 381.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 386.40 | 376.80 | 381.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 381.30 | 377.70 | 381.69 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 392.60 | 384.27 | 383.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 395.85 | 386.58 | 384.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 387.75 | 393.57 | 390.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 387.75 | 393.57 | 390.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 387.75 | 393.57 | 390.08 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 384.25 | 388.23 | 388.47 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 395.05 | 389.56 | 389.03 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 376.30 | 388.38 | 389.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 375.45 | 385.79 | 388.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 386.05 | 380.89 | 383.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 386.05 | 380.89 | 383.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 386.05 | 380.89 | 383.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 385.10 | 380.89 | 383.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 385.40 | 381.79 | 383.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 387.00 | 381.79 | 383.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 384.00 | 382.66 | 383.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 395.05 | 382.66 | 383.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 396.50 | 385.43 | 384.74 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 373.60 | 385.76 | 385.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 371.50 | 382.91 | 384.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 379.70 | 365.42 | 370.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 379.70 | 365.42 | 370.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 379.70 | 365.42 | 370.48 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 383.30 | 374.95 | 373.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 385.00 | 377.87 | 375.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 419.00 | 423.84 | 416.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 419.00 | 423.84 | 416.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 418.40 | 422.75 | 416.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 416.80 | 422.75 | 416.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 418.55 | 420.90 | 417.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 418.70 | 420.90 | 417.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 417.15 | 420.25 | 417.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 420.40 | 419.94 | 417.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 418.90 | 419.61 | 417.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 424.50 | 417.01 | 416.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 15:15:00 | 416.00 | 418.16 | 418.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 15:15:00 | 416.00 | 418.16 | 418.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 09:15:00 | 406.80 | 415.89 | 417.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 414.80 | 410.18 | 412.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 414.80 | 410.18 | 412.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 414.80 | 410.18 | 412.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 414.80 | 410.18 | 412.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 410.20 | 410.19 | 412.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 409.00 | 410.19 | 412.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:45:00 | 408.70 | 410.24 | 412.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 13:15:00 | 408.85 | 410.24 | 412.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 407.70 | 411.56 | 412.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 416.15 | 412.48 | 412.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 416.15 | 412.48 | 412.70 | SL hit (close>static) qty=1.00 sl=414.85 alert=retest2 |

### Cycle 208 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 421.00 | 414.18 | 413.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 423.50 | 416.05 | 414.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 503.65 | 506.65 | 491.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:15:00 | 498.40 | 506.65 | 491.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 497.80 | 505.87 | 501.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 497.80 | 505.87 | 501.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 496.25 | 503.95 | 501.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 499.65 | 503.95 | 501.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 489.00 | 500.96 | 500.04 | SL hit (close<static) qty=1.00 sl=495.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 488.25 | 498.42 | 498.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 482.50 | 495.23 | 497.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 482.75 | 482.50 | 488.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 482.75 | 482.50 | 488.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 492.00 | 484.00 | 488.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:45:00 | 491.85 | 484.00 | 488.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 488.45 | 484.89 | 488.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 487.50 | 484.89 | 488.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:00:00 | 484.60 | 485.25 | 487.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 485.85 | 486.17 | 487.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 503.40 | 490.52 | 489.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 503.40 | 490.52 | 489.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 14:15:00 | 506.15 | 493.64 | 491.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 561.65 | 562.30 | 549.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 561.65 | 562.30 | 549.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 560.70 | 561.41 | 555.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 560.70 | 561.41 | 555.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-23 11:15:00 | 277.90 | 2023-05-29 12:15:00 | 279.77 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2023-06-02 09:15:00 | 268.40 | 2023-06-05 09:15:00 | 276.35 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2023-06-02 14:00:00 | 270.23 | 2023-06-05 09:15:00 | 276.35 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-06-12 11:45:00 | 283.45 | 2023-06-21 11:15:00 | 287.00 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2023-06-30 12:15:00 | 286.90 | 2023-07-03 12:15:00 | 315.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-16 10:15:00 | 416.33 | 2023-08-17 14:15:00 | 409.88 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-08-16 11:45:00 | 415.45 | 2023-08-17 14:15:00 | 409.88 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-08-17 09:15:00 | 418.73 | 2023-08-17 14:15:00 | 409.88 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2023-08-18 09:15:00 | 416.98 | 2023-08-28 14:15:00 | 423.08 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2023-08-21 09:15:00 | 421.50 | 2023-08-28 14:15:00 | 423.08 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2023-08-31 09:15:00 | 432.65 | 2023-09-05 09:15:00 | 475.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-18 13:00:00 | 381.00 | 2023-09-21 12:15:00 | 361.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-20 10:15:00 | 381.05 | 2023-09-21 12:15:00 | 362.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 13:00:00 | 381.00 | 2023-09-22 12:15:00 | 363.45 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2023-09-20 10:15:00 | 381.05 | 2023-09-22 12:15:00 | 363.45 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-10-18 11:45:00 | 400.95 | 2023-10-20 12:15:00 | 404.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-10-20 10:45:00 | 401.50 | 2023-10-20 12:15:00 | 404.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-10-25 12:15:00 | 382.80 | 2023-10-26 09:15:00 | 363.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-25 12:15:00 | 382.80 | 2023-10-26 12:15:00 | 381.48 | STOP_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-06 09:15:00 | 466.00 | 2023-11-07 11:15:00 | 451.88 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2023-11-07 13:15:00 | 455.70 | 2023-11-08 09:15:00 | 443.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-11-07 13:45:00 | 455.40 | 2023-11-08 09:15:00 | 443.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2023-11-12 18:15:00 | 456.28 | 2023-11-13 09:15:00 | 449.38 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-11-23 14:15:00 | 441.43 | 2023-11-24 09:15:00 | 480.98 | STOP_HIT | 1.00 | -8.96% |
| SELL | retest2 | 2023-11-30 09:45:00 | 453.15 | 2023-11-30 11:15:00 | 465.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2023-11-30 10:15:00 | 454.53 | 2023-11-30 11:15:00 | 465.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2023-11-30 10:45:00 | 454.80 | 2023-11-30 11:15:00 | 465.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2023-12-04 09:15:00 | 477.33 | 2023-12-05 14:15:00 | 468.65 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2023-12-14 14:45:00 | 459.88 | 2023-12-20 13:15:00 | 436.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-15 10:45:00 | 459.38 | 2023-12-20 13:15:00 | 436.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-15 11:15:00 | 459.50 | 2023-12-20 13:15:00 | 436.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 10:15:00 | 459.90 | 2023-12-20 13:15:00 | 436.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-14 14:45:00 | 459.88 | 2023-12-22 09:15:00 | 447.85 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2023-12-15 10:45:00 | 459.38 | 2023-12-22 09:15:00 | 447.85 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2023-12-15 11:15:00 | 459.50 | 2023-12-22 09:15:00 | 447.85 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2023-12-20 10:15:00 | 459.90 | 2023-12-22 09:15:00 | 447.85 | STOP_HIT | 0.50 | 2.62% |
| BUY | retest2 | 2024-01-01 09:15:00 | 472.05 | 2024-01-02 09:15:00 | 464.68 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-01-01 09:45:00 | 470.05 | 2024-01-02 09:15:00 | 464.68 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-01-01 11:15:00 | 469.48 | 2024-01-02 09:15:00 | 464.68 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-01-01 13:30:00 | 469.35 | 2024-01-02 09:15:00 | 464.68 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-01-02 09:15:00 | 470.23 | 2024-01-02 09:15:00 | 464.68 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-01-08 09:15:00 | 480.95 | 2024-01-15 10:15:00 | 529.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-24 12:00:00 | 522.50 | 2024-01-24 14:15:00 | 530.23 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-01-29 09:15:00 | 534.50 | 2024-01-29 09:15:00 | 524.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-02-07 11:30:00 | 524.90 | 2024-02-09 13:15:00 | 498.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 09:30:00 | 523.60 | 2024-02-12 09:15:00 | 497.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:30:00 | 524.90 | 2024-02-13 14:15:00 | 502.43 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2024-02-08 09:30:00 | 523.60 | 2024-02-13 14:15:00 | 502.43 | STOP_HIT | 0.50 | 4.04% |
| BUY | retest2 | 2024-02-16 09:15:00 | 520.80 | 2024-02-16 11:15:00 | 510.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-02-21 10:15:00 | 509.03 | 2024-02-26 09:15:00 | 510.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-02-23 15:00:00 | 507.45 | 2024-02-26 09:15:00 | 510.95 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-03-11 09:15:00 | 462.50 | 2024-03-12 11:15:00 | 439.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 462.50 | 2024-03-13 10:15:00 | 416.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-20 15:00:00 | 420.03 | 2024-03-21 09:15:00 | 440.00 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2024-04-01 09:15:00 | 485.18 | 2024-04-12 09:15:00 | 497.90 | STOP_HIT | 1.00 | 2.62% |
| SELL | retest2 | 2024-04-15 14:45:00 | 495.13 | 2024-04-16 09:15:00 | 509.63 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-05-02 09:15:00 | 574.00 | 2024-05-02 13:15:00 | 564.83 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-05-02 13:30:00 | 569.30 | 2024-05-02 14:15:00 | 568.92 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-05-09 09:15:00 | 525.00 | 2024-05-10 09:15:00 | 472.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-22 15:00:00 | 574.95 | 2024-05-28 14:15:00 | 570.23 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-05-23 09:30:00 | 574.98 | 2024-05-28 15:15:00 | 568.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-23 15:15:00 | 573.88 | 2024-05-28 15:15:00 | 568.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-05-24 09:30:00 | 576.17 | 2024-05-28 15:15:00 | 568.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-27 09:15:00 | 582.55 | 2024-05-28 15:15:00 | 568.50 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-06-06 12:45:00 | 520.73 | 2024-06-07 09:15:00 | 536.92 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-06-19 14:45:00 | 655.15 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2024-06-20 09:15:00 | 651.98 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2024-06-20 10:30:00 | 664.88 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-06-20 15:00:00 | 652.45 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2024-06-21 15:15:00 | 656.95 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-06-24 10:00:00 | 670.20 | 2024-06-27 14:15:00 | 661.63 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-04 15:15:00 | 675.00 | 2024-07-08 09:15:00 | 669.55 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-07-24 09:30:00 | 624.45 | 2024-07-26 12:15:00 | 628.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-07-24 10:15:00 | 624.90 | 2024-07-26 12:15:00 | 628.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-07-26 09:45:00 | 626.00 | 2024-07-26 12:15:00 | 628.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-08-02 10:30:00 | 636.50 | 2024-08-02 12:15:00 | 628.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest1 | 2024-08-06 14:15:00 | 568.60 | 2024-08-07 11:15:00 | 594.95 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2024-08-08 09:30:00 | 590.90 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-08-08 10:15:00 | 591.80 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-08-08 11:00:00 | 590.75 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-08-08 11:30:00 | 591.60 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-08-09 15:00:00 | 581.95 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-08-12 11:30:00 | 581.95 | 2024-08-12 14:15:00 | 596.10 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-08-14 11:45:00 | 591.70 | 2024-09-06 15:15:00 | 619.70 | STOP_HIT | 1.00 | 4.73% |
| BUY | retest2 | 2024-08-14 12:15:00 | 591.05 | 2024-09-06 15:15:00 | 619.70 | STOP_HIT | 1.00 | 4.85% |
| BUY | retest2 | 2024-08-14 12:45:00 | 591.15 | 2024-09-06 15:15:00 | 619.70 | STOP_HIT | 1.00 | 4.83% |
| BUY | retest2 | 2024-08-14 13:45:00 | 592.65 | 2024-09-06 15:15:00 | 619.70 | STOP_HIT | 1.00 | 4.56% |
| BUY | retest2 | 2024-08-16 09:15:00 | 600.60 | 2024-09-06 15:15:00 | 619.70 | STOP_HIT | 1.00 | 3.18% |
| SELL | retest2 | 2024-09-12 14:00:00 | 621.35 | 2024-09-13 09:15:00 | 639.70 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-10-03 12:30:00 | 705.25 | 2024-10-03 14:15:00 | 690.15 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-10-09 13:15:00 | 645.00 | 2024-10-09 15:15:00 | 656.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-10-29 09:30:00 | 556.80 | 2024-10-31 11:15:00 | 563.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-10-29 10:00:00 | 553.70 | 2024-10-31 11:15:00 | 563.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-10-31 10:00:00 | 556.95 | 2024-10-31 11:15:00 | 563.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-06 09:15:00 | 585.05 | 2024-11-08 11:15:00 | 577.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-06 09:45:00 | 583.95 | 2024-11-08 11:15:00 | 577.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-11-08 10:00:00 | 583.95 | 2024-11-08 11:15:00 | 577.30 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-11-14 09:15:00 | 537.30 | 2024-11-19 10:15:00 | 573.00 | STOP_HIT | 1.00 | -6.64% |
| SELL | retest2 | 2024-11-18 09:45:00 | 540.00 | 2024-11-19 10:15:00 | 573.00 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2024-11-18 10:30:00 | 538.00 | 2024-11-19 10:15:00 | 573.00 | STOP_HIT | 1.00 | -6.51% |
| BUY | retest2 | 2024-11-27 09:30:00 | 578.70 | 2024-11-29 12:15:00 | 574.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-12-16 09:15:00 | 650.55 | 2024-12-16 10:15:00 | 646.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-17 10:45:00 | 638.30 | 2024-12-19 09:15:00 | 606.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:45:00 | 638.30 | 2024-12-19 14:15:00 | 617.55 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-01-08 12:00:00 | 601.55 | 2025-01-13 11:15:00 | 571.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 597.15 | 2025-01-13 11:15:00 | 567.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 598.10 | 2025-01-13 11:15:00 | 568.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:00:00 | 601.55 | 2025-01-14 09:15:00 | 541.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 597.15 | 2025-01-14 09:15:00 | 537.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 598.10 | 2025-01-14 09:15:00 | 538.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-20 10:45:00 | 601.50 | 2025-01-22 10:15:00 | 586.15 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-01-29 14:30:00 | 528.70 | 2025-01-31 11:15:00 | 536.30 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-10 11:15:00 | 511.85 | 2025-02-10 13:15:00 | 503.85 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest1 | 2025-02-12 14:00:00 | 461.50 | 2025-02-13 09:15:00 | 478.45 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-02-21 14:45:00 | 475.30 | 2025-02-24 13:15:00 | 465.15 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-03-26 15:00:00 | 457.00 | 2025-04-01 10:15:00 | 445.50 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-03-27 10:15:00 | 457.40 | 2025-04-01 10:15:00 | 445.50 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-04-01 09:45:00 | 454.55 | 2025-04-01 10:15:00 | 445.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-04-08 10:30:00 | 417.90 | 2025-04-11 11:15:00 | 434.40 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-04-08 11:45:00 | 419.00 | 2025-04-11 11:15:00 | 434.40 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-04-08 13:30:00 | 418.50 | 2025-04-11 11:15:00 | 434.40 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-04-08 15:00:00 | 418.00 | 2025-04-11 11:15:00 | 434.40 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-04-28 10:15:00 | 542.65 | 2025-05-06 09:15:00 | 596.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 09:15:00 | 694.95 | 2025-05-26 11:15:00 | 673.75 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-05-29 09:15:00 | 684.05 | 2025-05-29 13:15:00 | 681.15 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-05-29 10:30:00 | 682.15 | 2025-05-29 13:15:00 | 681.15 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-05-29 13:15:00 | 682.35 | 2025-05-29 13:15:00 | 681.15 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-06-04 13:15:00 | 699.00 | 2025-06-06 09:15:00 | 688.65 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-05 12:30:00 | 696.35 | 2025-06-06 09:15:00 | 688.65 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-09 10:15:00 | 691.40 | 2025-06-10 10:15:00 | 699.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-06-09 13:30:00 | 690.55 | 2025-06-10 10:15:00 | 699.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-17 09:15:00 | 663.30 | 2025-06-19 10:15:00 | 630.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:00:00 | 664.65 | 2025-06-19 10:15:00 | 631.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 663.30 | 2025-06-20 09:15:00 | 640.50 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2025-06-17 11:00:00 | 664.65 | 2025-06-20 09:15:00 | 640.50 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-07-22 10:15:00 | 600.00 | 2025-07-28 09:15:00 | 570.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:30:00 | 599.00 | 2025-07-28 09:15:00 | 569.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 600.00 | 2025-07-29 09:15:00 | 569.75 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2025-07-23 09:30:00 | 599.00 | 2025-07-29 09:15:00 | 569.75 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-08-05 13:15:00 | 568.40 | 2025-08-05 14:15:00 | 574.15 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-12 11:00:00 | 539.00 | 2025-08-13 09:15:00 | 560.80 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2025-08-12 12:45:00 | 539.45 | 2025-08-13 09:15:00 | 560.80 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-08-29 13:45:00 | 547.20 | 2025-09-01 09:15:00 | 558.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-09-10 09:15:00 | 567.55 | 2025-09-16 10:15:00 | 624.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-09 09:15:00 | 614.65 | 2025-10-09 11:15:00 | 602.10 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-10-09 13:45:00 | 608.10 | 2025-10-09 14:15:00 | 603.35 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-16 12:00:00 | 543.80 | 2025-10-20 15:15:00 | 544.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-10-16 13:45:00 | 544.00 | 2025-10-20 15:15:00 | 544.95 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-20 11:00:00 | 543.25 | 2025-10-20 15:15:00 | 544.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-10-20 11:30:00 | 543.75 | 2025-10-20 15:15:00 | 544.95 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-10-28 09:15:00 | 563.40 | 2025-10-28 12:15:00 | 555.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-17 09:15:00 | 543.30 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-11-17 12:15:00 | 540.30 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-11-18 14:00:00 | 538.20 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-19 09:15:00 | 538.30 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-19 10:30:00 | 542.90 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-19 14:30:00 | 542.70 | 2025-11-20 10:15:00 | 533.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-27 09:15:00 | 498.85 | 2025-11-28 10:15:00 | 510.80 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-11-27 10:00:00 | 500.10 | 2025-11-28 10:15:00 | 510.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-12-04 13:15:00 | 483.40 | 2025-12-10 09:15:00 | 484.95 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-12-11 14:15:00 | 478.40 | 2025-12-17 09:15:00 | 478.35 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-12-11 15:00:00 | 479.00 | 2025-12-17 09:15:00 | 478.35 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-12-12 09:15:00 | 481.60 | 2025-12-17 09:15:00 | 478.35 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-19 13:00:00 | 475.90 | 2025-12-19 13:15:00 | 480.05 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-29 11:15:00 | 476.30 | 2025-12-31 15:15:00 | 481.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-29 12:15:00 | 477.25 | 2025-12-31 15:15:00 | 481.60 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 13:45:00 | 477.40 | 2025-12-31 15:15:00 | 481.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-29 14:30:00 | 475.80 | 2025-12-31 15:15:00 | 481.60 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-22 10:15:00 | 376.50 | 2026-01-28 11:15:00 | 382.80 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-23 10:15:00 | 376.85 | 2026-01-28 11:15:00 | 382.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-06 09:15:00 | 444.00 | 2026-02-09 11:15:00 | 488.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 430.55 | 2026-03-02 09:15:00 | 409.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 430.55 | 2026-03-05 14:15:00 | 398.90 | STOP_HIT | 0.50 | 7.35% |
| BUY | retest2 | 2026-04-13 10:45:00 | 420.40 | 2026-04-15 15:15:00 | 416.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-04-13 12:15:00 | 418.90 | 2026-04-15 15:15:00 | 416.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-04-15 09:15:00 | 424.50 | 2026-04-15 15:15:00 | 416.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-04-17 11:15:00 | 409.00 | 2026-04-20 10:15:00 | 416.15 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-04-17 12:45:00 | 408.70 | 2026-04-20 10:15:00 | 416.15 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-17 13:15:00 | 408.85 | 2026-04-20 10:15:00 | 416.15 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-04-20 09:30:00 | 407.70 | 2026-04-20 10:15:00 | 416.15 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-28 09:15:00 | 499.65 | 2026-04-28 09:15:00 | 489.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-04-29 15:15:00 | 487.50 | 2026-04-30 13:15:00 | 503.40 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-30 10:00:00 | 484.60 | 2026-04-30 13:15:00 | 503.40 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2026-04-30 11:30:00 | 485.85 | 2026-04-30 13:15:00 | 503.40 | STOP_HIT | 1.00 | -3.61% |
