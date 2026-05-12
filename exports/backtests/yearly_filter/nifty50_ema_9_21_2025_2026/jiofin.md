# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 239.98
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 49 |
| ALERT2 | 47 |
| ALERT2_SKIP | 18 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 65 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 80 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 51
- **Target hits / Stop hits / Partials:** 2 / 67 / 11
- **Avg / median % per leg:** 1.07% / -0.68%
- **Sum % (uncompounded):** 85.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 6 | 22.2% | 2 | 25 | 0 | 0.18% | 4.9% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 2.13% | 6.4% |
| BUY @ 3rd Alert (retest2) | 24 | 3 | 12.5% | 2 | 22 | 0 | -0.06% | -1.5% |
| SELL (all) | 53 | 23 | 43.4% | 0 | 42 | 11 | 1.53% | 81.0% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.02% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 22 | 42.3% | 0 | 41 | 11 | 1.56% | 80.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 1.60% | 6.4% |
| retest2 (combined) | 76 | 25 | 32.9% | 2 | 63 | 11 | 1.05% | 79.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 260.50 | 252.84 | 252.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 261.75 | 256.92 | 254.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 265.90 | 266.76 | 263.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 15:00:00 | 267.80 | 266.97 | 264.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:00:00 | 267.40 | 267.20 | 264.79 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:30:00 | 268.60 | 267.26 | 265.04 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 273.65 | 276.22 | 274.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 273.65 | 276.22 | 274.30 | SL hit (close<ema400) qty=1.00 sl=274.30 alert=retest1 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 271.60 | 273.73 | 273.89 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 274.20 | 273.98 | 273.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 275.85 | 274.44 | 274.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 273.75 | 274.69 | 274.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 274.60 | 274.67 | 274.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 273.45 | 274.67 | 274.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 274.65 | 274.67 | 274.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 276.60 | 274.67 | 274.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 286.30 | 287.29 | 287.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 286.30 | 287.29 | 287.35 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 288.70 | 287.33 | 287.32 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 286.25 | 287.25 | 287.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 284.85 | 286.47 | 286.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 284.90 | 284.58 | 285.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 284.90 | 284.58 | 285.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 286.15 | 284.89 | 285.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 285.60 | 284.89 | 285.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 288.40 | 285.59 | 285.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 288.40 | 285.59 | 285.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 290.10 | 286.49 | 286.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 292.20 | 289.50 | 287.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 302.95 | 303.08 | 300.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:45:00 | 302.70 | 303.08 | 300.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 301.00 | 302.54 | 300.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 300.05 | 302.54 | 300.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 300.75 | 302.18 | 300.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 300.70 | 302.18 | 300.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 305.30 | 302.81 | 300.81 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 295.15 | 300.30 | 300.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 292.60 | 297.51 | 299.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 294.60 | 293.88 | 295.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 294.60 | 293.88 | 295.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 293.65 | 293.89 | 295.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:45:00 | 290.80 | 292.96 | 294.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:15:00 | 290.90 | 292.96 | 294.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 290.50 | 292.16 | 293.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 291.00 | 291.42 | 292.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 288.30 | 286.20 | 288.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 286.70 | 286.20 | 288.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 291.85 | 287.33 | 288.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 291.85 | 287.33 | 288.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 292.60 | 288.38 | 288.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 290.55 | 288.38 | 288.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 293.25 | 289.94 | 289.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 293.25 | 289.94 | 289.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 294.25 | 290.80 | 289.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 293.15 | 291.85 | 290.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 293.20 | 291.85 | 290.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 10:15:00 | 322.46 | 313.39 | 307.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 324.45 | 325.50 | 325.61 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 328.40 | 326.10 | 325.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 331.40 | 328.52 | 327.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 329.75 | 330.95 | 329.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 329.70 | 330.70 | 329.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 330.40 | 330.70 | 329.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 332.50 | 331.06 | 329.69 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 326.65 | 330.00 | 330.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 325.40 | 327.40 | 328.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 321.40 | 321.28 | 323.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 321.40 | 321.28 | 323.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 319.75 | 320.61 | 322.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 321.10 | 320.61 | 322.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 319.30 | 319.80 | 320.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 320.50 | 319.80 | 320.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 315.60 | 318.58 | 319.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 314.55 | 318.58 | 319.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:00:00 | 314.60 | 316.89 | 317.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 318.05 | 315.14 | 315.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 318.05 | 315.14 | 315.02 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 311.50 | 315.29 | 315.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 311.25 | 313.23 | 314.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:15:00 | 315.00 | 312.62 | 313.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 314.05 | 312.91 | 313.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 312.55 | 312.67 | 313.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 312.75 | 310.66 | 312.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 317.05 | 312.65 | 312.73 | SL hit (close>static) qty=1.00 sl=315.85 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 317.50 | 313.62 | 313.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 318.95 | 314.68 | 313.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 320.00 | 320.48 | 318.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 327.80 | 320.48 | 318.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 328.75 | 322.13 | 319.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 332.15 | 328.75 | 325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 334.10 | 329.36 | 326.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 331.80 | 330.55 | 327.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 332.10 | 330.67 | 328.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 329.30 | 332.21 | 331.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 329.30 | 332.21 | 331.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 327.50 | 331.27 | 330.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 327.50 | 331.27 | 330.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 327.95 | 330.60 | 330.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 327.95 | 330.60 | 330.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 321.90 | 327.16 | 328.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 324.70 | 323.22 | 325.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 324.70 | 323.22 | 325.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 327.40 | 324.06 | 326.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 324.05 | 324.06 | 326.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 328.60 | 324.52 | 324.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 328.60 | 324.52 | 324.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 329.95 | 326.09 | 325.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 327.10 | 327.64 | 326.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 327.10 | 327.64 | 326.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 327.15 | 329.51 | 328.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 327.15 | 329.51 | 328.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 326.95 | 329.00 | 328.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 326.95 | 329.00 | 328.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 328.55 | 328.56 | 328.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 331.25 | 328.10 | 328.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:15:00 | 328.95 | 328.76 | 328.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 329.75 | 328.96 | 328.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:30:00 | 329.10 | 330.26 | 329.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 329.85 | 330.18 | 329.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 329.85 | 330.18 | 329.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 329.90 | 330.12 | 329.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 329.65 | 330.12 | 329.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 329.25 | 329.95 | 329.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 329.25 | 329.95 | 329.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 329.35 | 329.83 | 329.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 328.10 | 329.48 | 329.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 328.10 | 329.48 | 329.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 327.10 | 328.66 | 329.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 321.75 | 321.08 | 323.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 321.75 | 321.08 | 323.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 311.50 | 312.93 | 314.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 311.40 | 312.93 | 314.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 311.35 | 312.62 | 313.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 311.20 | 312.08 | 312.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 311.40 | 312.47 | 312.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 311.85 | 311.08 | 311.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 311.85 | 311.08 | 311.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 310.90 | 311.04 | 311.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 310.30 | 310.80 | 311.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:00:00 | 310.20 | 310.57 | 311.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 312.40 | 311.59 | 311.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 312.40 | 311.59 | 311.49 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 310.85 | 311.34 | 311.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 310.50 | 311.05 | 311.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 312.80 | 309.92 | 310.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 312.25 | 310.39 | 310.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 312.25 | 310.39 | 310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 312.80 | 310.87 | 310.77 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 310.30 | 310.74 | 310.76 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 311.35 | 310.86 | 310.81 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 310.40 | 310.85 | 310.91 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 314.80 | 311.64 | 311.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 315.80 | 313.76 | 312.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 12:15:00 | 316.50 | 316.68 | 315.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 13:00:00 | 316.50 | 316.68 | 315.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 315.50 | 316.44 | 315.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 315.50 | 316.44 | 315.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 315.60 | 316.27 | 315.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 316.65 | 316.20 | 315.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 317.30 | 317.15 | 316.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 313.90 | 316.35 | 316.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 313.90 | 316.35 | 316.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 311.80 | 315.06 | 315.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 298.60 | 298.37 | 301.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:30:00 | 297.85 | 298.19 | 301.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 297.80 | 295.27 | 296.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 297.80 | 295.27 | 296.70 | SL hit (close>ema400) qty=1.00 sl=296.70 alert=retest1 |

### Cycle 27 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 300.75 | 297.86 | 297.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 301.00 | 298.96 | 298.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 308.90 | 309.55 | 306.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 308.05 | 309.55 | 306.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 306.05 | 308.85 | 306.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 306.05 | 308.85 | 306.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 305.70 | 308.22 | 306.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 305.70 | 308.22 | 306.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 306.15 | 307.51 | 306.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 306.15 | 307.51 | 306.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 304.40 | 306.88 | 306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 304.60 | 306.88 | 306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 307.55 | 306.62 | 306.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 306.45 | 306.62 | 306.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 306.50 | 306.80 | 306.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:30:00 | 306.60 | 306.80 | 306.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 307.45 | 306.93 | 306.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 306.65 | 306.93 | 306.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 309.15 | 307.40 | 306.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 309.85 | 307.89 | 307.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 309.60 | 308.55 | 307.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 306.20 | 308.09 | 307.69 | SL hit (close<static) qty=1.00 sl=306.65 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 305.80 | 307.17 | 307.32 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 308.65 | 307.60 | 307.46 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 306.30 | 307.33 | 307.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 305.75 | 307.02 | 307.23 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 312.10 | 307.48 | 307.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 312.95 | 310.60 | 309.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 311.70 | 311.80 | 310.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 311.70 | 311.80 | 310.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 309.10 | 311.31 | 310.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 308.95 | 311.31 | 310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 309.50 | 310.95 | 310.47 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 308.15 | 310.18 | 310.19 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 314.70 | 310.62 | 310.32 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 308.65 | 310.80 | 311.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 308.45 | 309.89 | 310.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 307.75 | 307.67 | 308.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:00:00 | 307.75 | 307.67 | 308.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 305.85 | 306.24 | 307.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 307.45 | 306.24 | 307.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 307.40 | 306.25 | 307.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 307.25 | 306.25 | 307.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 307.20 | 306.44 | 307.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:15:00 | 306.45 | 306.44 | 307.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 308.40 | 306.83 | 307.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 308.40 | 306.83 | 307.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 308.40 | 307.14 | 307.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 309.50 | 307.14 | 307.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 309.25 | 307.77 | 307.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 311.00 | 309.51 | 308.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 309.55 | 310.02 | 309.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 310.00 | 310.02 | 309.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 309.45 | 310.02 | 309.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 310.10 | 309.98 | 309.41 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 306.60 | 308.82 | 309.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 306.00 | 307.62 | 308.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 298.70 | 298.51 | 300.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 298.70 | 298.51 | 300.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 300.95 | 299.32 | 300.49 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 302.35 | 301.04 | 301.02 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 298.65 | 300.78 | 300.93 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 303.65 | 301.28 | 301.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 304.80 | 302.47 | 301.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 311.05 | 311.51 | 308.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 311.05 | 311.51 | 308.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 312.75 | 311.60 | 309.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 314.30 | 311.60 | 309.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 315.35 | 312.39 | 309.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 314.30 | 312.77 | 310.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 313.90 | 313.00 | 310.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 312.35 | 313.36 | 312.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 312.30 | 313.36 | 312.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 312.10 | 313.11 | 312.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 311.70 | 313.11 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 312.45 | 312.98 | 312.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 310.40 | 312.98 | 312.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 308.40 | 312.06 | 311.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 308.40 | 312.06 | 311.92 | SL hit (close<static) qty=1.00 sl=309.30 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 308.65 | 311.38 | 311.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 306.35 | 309.10 | 310.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 306.20 | 305.92 | 307.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 307.45 | 306.23 | 307.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 307.80 | 306.23 | 307.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 307.70 | 306.52 | 307.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 308.05 | 306.52 | 307.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 308.15 | 306.85 | 307.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:30:00 | 308.30 | 306.85 | 307.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 304.40 | 302.12 | 302.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 304.40 | 302.12 | 302.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 305.15 | 302.73 | 303.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 305.25 | 302.73 | 303.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 308.15 | 304.31 | 303.82 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 304.90 | 305.84 | 305.91 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 306.25 | 305.82 | 305.80 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 305.10 | 305.68 | 305.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 301.45 | 304.76 | 305.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 302.70 | 302.14 | 303.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 302.70 | 302.14 | 303.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 301.55 | 302.02 | 302.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 300.90 | 301.86 | 302.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 300.70 | 302.02 | 302.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 301.30 | 302.02 | 302.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 303.85 | 302.69 | 302.76 | SL hit (close>static) qty=1.00 sl=303.45 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 303.85 | 302.92 | 302.86 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 300.90 | 302.52 | 302.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 298.05 | 301.29 | 302.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 293.35 | 292.42 | 294.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 293.35 | 292.42 | 294.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 294.70 | 292.88 | 294.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 295.85 | 292.88 | 294.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 296.45 | 293.59 | 294.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 296.45 | 293.59 | 294.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 298.30 | 295.43 | 295.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 300.80 | 297.00 | 296.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 297.75 | 299.32 | 298.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 298.50 | 299.16 | 298.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 298.25 | 299.16 | 298.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 299.50 | 299.22 | 298.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 298.10 | 299.22 | 298.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 295.95 | 298.75 | 298.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 295.95 | 298.75 | 298.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 295.20 | 298.04 | 298.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 294.85 | 297.40 | 297.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 296.50 | 293.03 | 293.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 295.10 | 293.45 | 293.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 295.80 | 293.45 | 293.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 296.35 | 294.37 | 294.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 297.10 | 294.92 | 294.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 298.60 | 298.87 | 297.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 298.80 | 298.87 | 297.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 298.90 | 299.78 | 298.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 298.90 | 299.78 | 298.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 298.60 | 299.54 | 298.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 298.55 | 299.54 | 298.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 298.60 | 299.35 | 298.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 299.55 | 299.28 | 298.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 297.20 | 298.68 | 298.65 | SL hit (close<static) qty=1.00 sl=298.05 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 297.30 | 298.41 | 298.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 295.55 | 297.47 | 298.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 293.20 | 293.13 | 294.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 294.55 | 293.13 | 294.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 294.20 | 293.35 | 294.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 294.10 | 293.35 | 294.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 295.55 | 293.79 | 294.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 295.55 | 293.79 | 294.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 296.00 | 294.23 | 294.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 295.55 | 294.23 | 294.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 295.00 | 294.96 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 295.85 | 294.96 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 294.95 | 294.95 | 294.99 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 295.50 | 295.06 | 295.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 295.65 | 295.19 | 295.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 299.50 | 300.45 | 299.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 300.40 | 300.44 | 299.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 300.90 | 300.45 | 299.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 294.65 | 299.15 | 299.06 | SL hit (close<static) qty=1.00 sl=299.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 295.65 | 298.45 | 298.75 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 304.30 | 299.71 | 299.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 11:15:00 | 304.75 | 300.72 | 299.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 301.25 | 302.28 | 301.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 296.50 | 301.13 | 300.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 296.50 | 301.13 | 300.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 296.55 | 300.21 | 300.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 294.35 | 298.27 | 299.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 287.80 | 287.44 | 291.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 287.80 | 287.44 | 291.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 285.80 | 287.35 | 290.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 284.30 | 286.77 | 289.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 284.15 | 286.15 | 288.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 284.10 | 285.73 | 288.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 283.95 | 285.19 | 288.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 286.85 | 285.36 | 287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 286.85 | 285.36 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 286.75 | 285.64 | 287.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 287.20 | 285.64 | 287.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 287.65 | 286.04 | 287.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 287.35 | 286.04 | 287.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 285.90 | 286.01 | 287.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 285.50 | 286.01 | 287.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 284.90 | 286.08 | 286.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 271.22 | 276.08 | 279.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 270.65 | 276.08 | 279.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 270.08 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.94 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.89 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.75 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 266.75 | 265.70 | 269.42 | SL hit (close>ema200) qty=0.50 sl=265.70 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 263.60 | 250.37 | 249.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 264.90 | 253.27 | 250.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 266.90 | 268.59 | 265.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 266.90 | 268.59 | 265.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 268.85 | 270.53 | 269.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 269.00 | 270.53 | 269.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 270.05 | 270.44 | 269.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 270.65 | 270.44 | 269.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 270.65 | 270.25 | 269.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 267.85 | 269.76 | 269.72 | SL hit (close<static) qty=1.00 sl=268.65 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 268.00 | 269.41 | 269.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 267.50 | 269.03 | 269.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 262.60 | 262.20 | 264.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:45:00 | 262.60 | 262.20 | 264.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 262.55 | 262.30 | 263.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 263.20 | 262.30 | 263.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 262.20 | 262.47 | 263.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 263.35 | 262.47 | 263.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 262.65 | 262.39 | 263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 264.15 | 262.39 | 263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 263.25 | 262.57 | 263.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 263.25 | 262.57 | 263.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 263.40 | 262.73 | 263.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:00:00 | 263.40 | 262.73 | 263.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 263.80 | 262.95 | 263.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:30:00 | 263.90 | 262.95 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 264.40 | 263.35 | 263.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 264.40 | 263.35 | 263.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 263.90 | 263.46 | 263.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 263.25 | 263.46 | 263.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 257.80 | 258.96 | 260.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 257.15 | 258.96 | 260.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 255.35 | 258.14 | 259.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 257.10 | 256.23 | 257.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 257.20 | 257.12 | 257.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 256.35 | 256.97 | 257.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 255.50 | 256.85 | 257.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 255.70 | 256.36 | 256.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 258.05 | 256.70 | 256.95 | SL hit (close>static) qty=1.00 sl=257.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 241.80 | 236.66 | 236.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 242.95 | 239.48 | 238.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 237.65 | 239.90 | 238.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 237.55 | 239.43 | 238.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 237.55 | 239.43 | 238.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 238.60 | 239.27 | 238.70 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 235.40 | 237.86 | 238.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 232.40 | 235.76 | 236.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 235.15 | 234.03 | 235.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 236.90 | 234.52 | 235.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 236.25 | 234.52 | 235.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 236.40 | 234.90 | 235.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 235.20 | 234.90 | 235.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 234.85 | 235.20 | 235.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 238.05 | 236.09 | 236.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 238.05 | 236.09 | 236.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 243.30 | 238.02 | 236.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 239.15 | 241.26 | 241.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 230.70 | 238.83 | 240.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 230.75 | 229.81 | 233.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 230.75 | 229.81 | 233.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 233.60 | 230.57 | 233.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 233.25 | 230.57 | 233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 232.90 | 231.03 | 233.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 233.55 | 231.03 | 233.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 232.60 | 231.35 | 233.16 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 240.35 | 234.58 | 234.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 241.00 | 235.86 | 234.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 233.00 | 236.32 | 235.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 233.10 | 235.67 | 235.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 232.90 | 235.67 | 235.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 233.25 | 235.19 | 235.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 229.35 | 232.75 | 233.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 228.61 | 227.62 | 230.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 226.32 | 230.46 | 230.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 227.74 | 228.07 | 228.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 226.97 | 228.45 | 229.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 228.54 | 228.46 | 229.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 233.11 | 229.41 | 229.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 233.11 | 229.41 | 229.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 235.70 | 231.31 | 230.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 242.70 | 243.43 | 239.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 242.70 | 243.43 | 239.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 238.37 | 241.99 | 239.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:00:00 | 238.37 | 241.99 | 239.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 238.97 | 241.38 | 239.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 238.92 | 241.38 | 239.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 237.22 | 241.70 | 241.10 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 237.98 | 240.36 | 240.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 237.42 | 239.77 | 240.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 243.75 | 239.73 | 239.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 242.45 | 240.28 | 240.22 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 239.82 | 240.94 | 241.01 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 241.51 | 241.11 | 241.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 243.03 | 241.49 | 241.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 236.49 | 241.67 | 241.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 238.06 | 240.95 | 241.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 236.60 | 238.75 | 239.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 236.08 | 235.69 | 237.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:45:00 | 236.31 | 235.69 | 237.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 239.71 | 236.50 | 237.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 239.71 | 236.50 | 237.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 240.00 | 237.20 | 237.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:45:00 | 240.00 | 237.20 | 237.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 245.88 | 239.45 | 238.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 249.23 | 243.11 | 240.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 244.20 | 245.93 | 243.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 244.20 | 245.93 | 243.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 251.42 | 252.87 | 251.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 251.42 | 252.87 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 250.62 | 252.42 | 251.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 250.62 | 252.42 | 251.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 250.70 | 252.07 | 251.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 245.34 | 252.07 | 251.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 243.56 | 249.06 | 249.76 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 251.55 | 249.32 | 249.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 253.70 | 250.97 | 250.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 250.44 | 250.87 | 250.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 250.44 | 250.87 | 250.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 248.33 | 250.36 | 249.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 247.61 | 250.36 | 249.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 249.15 | 250.12 | 249.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:00:00 | 249.65 | 250.02 | 249.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 248.17 | 249.65 | 249.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 248.17 | 249.65 | 249.66 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 251.12 | 249.67 | 249.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 252.70 | 251.09 | 250.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 251.06 | 251.16 | 250.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 251.06 | 251.16 | 250.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 252.29 | 251.39 | 250.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 250.65 | 251.39 | 250.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 251.40 | 251.49 | 250.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 251.42 | 251.49 | 250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 250.92 | 251.38 | 250.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:45:00 | 250.57 | 251.38 | 250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 251.15 | 251.33 | 250.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 249.89 | 251.33 | 250.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 250.30 | 251.12 | 250.87 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 249.80 | 250.67 | 250.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 248.91 | 250.32 | 250.53 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 15:00:00 | 267.80 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest1 | 2025-05-15 10:00:00 | 267.40 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest1 | 2025-05-15 10:30:00 | 268.60 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-05-23 09:15:00 | 276.60 | 2025-05-30 14:15:00 | 286.30 | STOP_HIT | 1.00 | 3.51% |
| SELL | retest2 | 2025-06-17 12:45:00 | 290.80 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-17 13:15:00 | 290.90 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-17 14:30:00 | 290.50 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-06-18 10:30:00 | 291.00 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-20 12:15:00 | 290.55 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-06-23 11:30:00 | 293.15 | 2025-06-27 10:15:00 | 322.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 12:00:00 | 293.20 | 2025-06-27 10:15:00 | 322.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 314.55 | 2025-07-24 10:15:00 | 318.05 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-22 10:00:00 | 314.60 | 2025-07-24 10:15:00 | 318.05 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-28 12:45:00 | 312.55 | 2025-07-29 11:15:00 | 317.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-29 10:00:00 | 312.75 | 2025-07-29 11:15:00 | 317.05 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-01 11:45:00 | 332.15 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-04 09:15:00 | 334.10 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-04 10:45:00 | 331.80 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-04 12:15:00 | 332.10 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-08-08 09:15:00 | 324.05 | 2025-08-11 14:15:00 | 328.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-18 09:15:00 | 331.25 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-18 12:15:00 | 328.95 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-18 13:00:00 | 329.75 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-08-20 09:30:00 | 329.10 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-29 15:15:00 | 311.40 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-03 10:15:00 | 311.35 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-09-03 12:45:00 | 311.20 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-04 12:15:00 | 311.40 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-05 11:30:00 | 310.30 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-05 14:00:00 | 310.20 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-09-18 09:15:00 | 316.65 | 2025-09-22 14:15:00 | 313.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-19 15:00:00 | 317.30 | 2025-09-22 14:15:00 | 313.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2025-09-29 11:30:00 | 297.85 | 2025-10-01 11:15:00 | 297.80 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-10-10 11:00:00 | 309.85 | 2025-10-13 09:15:00 | 306.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-10 12:30:00 | 309.60 | 2025-10-13 09:15:00 | 306.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-14 10:15:00 | 314.30 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-14 10:45:00 | 315.35 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-14 12:00:00 | 314.30 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-14 13:00:00 | 313.90 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-04 13:30:00 | 300.90 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-12-05 09:30:00 | 300.70 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-05 10:15:00 | 301.30 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-26 09:15:00 | 299.55 | 2025-12-26 12:15:00 | 297.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-01-06 09:30:00 | 300.90 | 2026-01-06 11:15:00 | 294.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-01-13 10:45:00 | 284.30 | 2026-01-20 09:15:00 | 271.22 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-13 11:30:00 | 284.15 | 2026-01-20 09:15:00 | 270.65 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-13 12:30:00 | 284.10 | 2026-01-20 11:15:00 | 270.08 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-13 13:45:00 | 283.95 | 2026-01-20 11:15:00 | 269.94 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-14 14:15:00 | 285.50 | 2026-01-20 11:15:00 | 269.89 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2026-01-16 09:30:00 | 284.90 | 2026-01-20 11:15:00 | 269.75 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2026-01-13 10:45:00 | 284.30 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.17% |
| SELL | retest2 | 2026-01-13 11:30:00 | 284.15 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.12% |
| SELL | retest2 | 2026-01-13 12:30:00 | 284.10 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2026-01-13 13:45:00 | 283.95 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2026-01-14 14:15:00 | 285.50 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2026-01-16 09:30:00 | 284.90 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.37% |
| BUY | retest2 | 2026-02-11 11:15:00 | 270.65 | 2026-02-12 09:15:00 | 267.85 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-11 13:15:00 | 270.65 | 2026-02-12 09:15:00 | 267.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-02-23 10:15:00 | 257.15 | 2026-02-27 10:15:00 | 258.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-24 09:15:00 | 255.35 | 2026-02-27 10:15:00 | 258.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-25 09:30:00 | 257.10 | 2026-02-27 13:15:00 | 257.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-02-25 12:45:00 | 257.20 | 2026-03-02 13:15:00 | 244.29 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-02-26 14:15:00 | 255.50 | 2026-03-02 13:15:00 | 244.25 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2026-02-27 09:45:00 | 255.70 | 2026-03-02 13:15:00 | 244.34 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-27 10:45:00 | 255.40 | 2026-03-04 09:15:00 | 242.58 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-02-27 15:00:00 | 255.00 | 2026-03-04 09:15:00 | 242.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 257.20 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2026-02-26 14:15:00 | 255.50 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2026-02-27 09:45:00 | 255.70 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-27 10:45:00 | 255.40 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2026-02-27 15:00:00 | 255.00 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-03-10 11:15:00 | 232.30 | 2026-03-11 09:15:00 | 238.90 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-10 11:45:00 | 232.25 | 2026-03-11 09:15:00 | 238.90 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-03-17 11:15:00 | 235.20 | 2026-03-17 13:15:00 | 238.05 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-17 12:15:00 | 234.85 | 2026-03-17 13:15:00 | 238.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-01 10:15:00 | 228.61 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-02 09:15:00 | 226.32 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-02 14:30:00 | 227.74 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-04-06 09:15:00 | 226.97 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-05-05 13:00:00 | 249.65 | 2026-05-05 13:15:00 | 248.17 | STOP_HIT | 1.00 | -0.59% |
