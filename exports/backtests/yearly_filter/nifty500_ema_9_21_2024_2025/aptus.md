# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 55 |
| ALERT3 | 249 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 113 |
| PARTIAL | 31 |
| TARGET_HIT | 3 |
| STOP_HIT | 122 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 77
- **Target hits / Stop hits / Partials:** 3 / 115 / 31
- **Avg / median % per leg:** 1.38% / -0.34%
- **Sum % (uncompounded):** 205.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 5 | 11.4% | 0 | 44 | 0 | -1.38% | -60.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 5 | 11.4% | 0 | 44 | 0 | -1.38% | -60.8% |
| SELL (all) | 105 | 67 | 63.8% | 3 | 71 | 31 | 2.53% | 265.9% |
| SELL @ 2nd Alert (retest1) | 9 | 9 | 100.0% | 0 | 5 | 4 | 4.26% | 38.3% |
| SELL @ 3rd Alert (retest2) | 96 | 58 | 60.4% | 3 | 66 | 27 | 2.37% | 227.6% |
| retest1 (combined) | 9 | 9 | 100.0% | 0 | 5 | 4 | 4.26% | 38.3% |
| retest2 (combined) | 140 | 63 | 45.0% | 3 | 110 | 27 | 1.19% | 166.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 319.00 | 318.04 | 318.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 320.30 | 318.49 | 318.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 15:15:00 | 317.25 | 318.25 | 318.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 15:15:00 | 317.25 | 318.25 | 318.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 317.25 | 318.25 | 318.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 317.25 | 318.25 | 318.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 316.40 | 317.88 | 317.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 313.65 | 316.37 | 317.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 313.35 | 312.94 | 314.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 10:15:00 | 313.35 | 312.94 | 314.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 313.35 | 312.94 | 314.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:45:00 | 313.65 | 312.94 | 314.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 316.30 | 313.20 | 313.92 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 12:15:00 | 317.00 | 314.41 | 314.37 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 305.95 | 312.72 | 313.61 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 313.70 | 308.14 | 308.03 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 306.50 | 308.71 | 308.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 305.45 | 307.26 | 308.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 10:15:00 | 306.05 | 305.98 | 307.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 10:15:00 | 306.05 | 305.98 | 307.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 306.05 | 305.98 | 307.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 306.55 | 305.98 | 307.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 307.40 | 306.27 | 307.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 307.55 | 306.27 | 307.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 306.15 | 306.24 | 307.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:30:00 | 305.85 | 306.24 | 307.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 304.65 | 305.86 | 306.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:30:00 | 306.15 | 305.86 | 306.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 300.90 | 304.73 | 306.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 305.15 | 304.73 | 306.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 311.00 | 302.92 | 303.08 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 313.75 | 305.09 | 304.05 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 290.00 | 306.00 | 306.08 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 313.75 | 305.31 | 304.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 316.50 | 311.40 | 308.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 317.75 | 318.11 | 315.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 317.75 | 318.11 | 315.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 346.80 | 346.09 | 341.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:00:00 | 349.20 | 345.45 | 343.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 349.55 | 346.40 | 343.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:15:00 | 348.80 | 346.58 | 344.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 339.45 | 345.16 | 345.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 339.45 | 345.16 | 345.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 338.35 | 342.40 | 343.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 14:15:00 | 340.30 | 338.19 | 340.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 340.30 | 338.19 | 340.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 340.30 | 338.19 | 340.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:30:00 | 340.15 | 338.19 | 340.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 340.70 | 338.69 | 340.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 339.45 | 338.69 | 340.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 336.90 | 338.33 | 339.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 335.35 | 338.33 | 339.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 341.80 | 331.36 | 330.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 341.80 | 331.36 | 330.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 344.50 | 335.82 | 333.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 340.50 | 341.29 | 337.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 340.50 | 341.29 | 337.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 335.95 | 339.75 | 338.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 335.95 | 339.75 | 338.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 337.00 | 339.20 | 337.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 344.35 | 339.20 | 337.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 14:15:00 | 339.85 | 341.80 | 342.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 14:15:00 | 339.85 | 341.80 | 342.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 334.00 | 340.19 | 341.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 337.40 | 336.32 | 338.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 337.40 | 336.32 | 338.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 337.40 | 336.32 | 338.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 337.40 | 336.32 | 338.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 334.60 | 327.26 | 329.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 334.60 | 327.26 | 329.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 333.10 | 328.43 | 330.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:15:00 | 331.65 | 329.20 | 330.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 325.00 | 323.72 | 323.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 325.00 | 323.72 | 323.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 14:15:00 | 325.75 | 324.41 | 324.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 323.40 | 324.54 | 324.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 323.40 | 324.54 | 324.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 323.40 | 324.54 | 324.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 322.40 | 324.54 | 324.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 324.60 | 324.55 | 324.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 322.00 | 324.55 | 324.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 323.45 | 324.33 | 324.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:15:00 | 315.95 | 324.33 | 324.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 316.95 | 322.86 | 323.49 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 326.65 | 322.25 | 321.97 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 319.55 | 322.10 | 322.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 318.00 | 320.11 | 320.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 317.55 | 317.49 | 318.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 317.55 | 317.49 | 318.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 317.55 | 317.49 | 318.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 11:45:00 | 315.90 | 317.05 | 318.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:30:00 | 315.60 | 316.94 | 318.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 13:45:00 | 316.00 | 316.74 | 318.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 14:30:00 | 315.75 | 316.52 | 317.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 318.75 | 316.99 | 317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 318.75 | 316.99 | 317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 317.30 | 317.05 | 317.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:15:00 | 316.60 | 317.09 | 317.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:30:00 | 315.60 | 316.62 | 317.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 312.45 | 315.98 | 316.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 300.77 | 303.39 | 305.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 300.10 | 302.61 | 304.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 299.82 | 302.61 | 304.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 300.20 | 302.61 | 304.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 299.96 | 302.61 | 304.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 299.82 | 302.61 | 304.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 303.05 | 302.29 | 303.66 | SL hit (close>ema200) qty=0.50 sl=302.29 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 14:15:00 | 304.50 | 303.97 | 303.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 306.90 | 304.80 | 304.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 12:15:00 | 304.80 | 304.87 | 304.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 12:15:00 | 304.80 | 304.87 | 304.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 304.80 | 304.87 | 304.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 304.80 | 304.87 | 304.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 315.75 | 312.75 | 310.08 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 309.00 | 311.02 | 311.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 308.05 | 310.43 | 310.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 313.50 | 310.67 | 310.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 12:15:00 | 313.50 | 310.67 | 310.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 313.50 | 310.67 | 310.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:30:00 | 314.25 | 310.67 | 310.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 311.85 | 310.90 | 311.00 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 311.75 | 311.07 | 311.07 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 310.00 | 310.89 | 311.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 308.60 | 310.43 | 310.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 314.00 | 309.99 | 310.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 314.00 | 309.99 | 310.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 314.00 | 309.99 | 310.28 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 315.00 | 310.99 | 310.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 14:15:00 | 318.20 | 313.85 | 312.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 09:15:00 | 318.30 | 320.26 | 317.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 318.30 | 320.26 | 317.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 318.30 | 320.26 | 317.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 318.10 | 320.26 | 317.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 321.80 | 320.57 | 317.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:30:00 | 322.80 | 321.16 | 318.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 331.35 | 323.20 | 320.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:30:00 | 326.80 | 321.33 | 321.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 319.80 | 321.02 | 321.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 319.80 | 321.02 | 321.18 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 13:15:00 | 323.00 | 321.50 | 321.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 324.60 | 322.56 | 321.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 323.80 | 325.20 | 323.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 323.80 | 325.20 | 323.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 323.80 | 325.20 | 323.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 323.80 | 325.20 | 323.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 329.05 | 325.97 | 324.28 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 318.25 | 323.52 | 324.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 316.45 | 322.11 | 323.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 324.45 | 319.19 | 321.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 324.45 | 319.19 | 321.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 324.45 | 319.19 | 321.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 324.45 | 319.19 | 321.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 323.05 | 319.96 | 321.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:30:00 | 322.60 | 320.46 | 321.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:15:00 | 322.25 | 321.46 | 321.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 324.50 | 322.09 | 321.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 324.50 | 322.09 | 321.86 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 320.65 | 321.70 | 321.81 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 323.00 | 321.89 | 321.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 333.70 | 324.78 | 323.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 332.05 | 333.82 | 330.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 12:00:00 | 332.05 | 333.82 | 330.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 332.70 | 333.16 | 331.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 330.50 | 333.16 | 331.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 328.65 | 332.26 | 330.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 328.90 | 332.26 | 330.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 326.95 | 331.20 | 330.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 326.95 | 331.20 | 330.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 362.50 | 365.86 | 361.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 362.50 | 365.86 | 361.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 362.00 | 365.09 | 361.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 367.50 | 365.09 | 361.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:30:00 | 364.25 | 364.46 | 362.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 15:15:00 | 360.00 | 363.20 | 363.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 360.00 | 363.20 | 363.44 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 365.00 | 363.19 | 363.18 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 355.20 | 361.59 | 362.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 348.35 | 355.49 | 358.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 352.35 | 352.13 | 355.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 352.35 | 352.13 | 355.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 353.85 | 346.23 | 349.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 356.00 | 346.23 | 349.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 357.40 | 348.47 | 350.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 356.35 | 348.47 | 350.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 12:15:00 | 361.35 | 352.73 | 351.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-07 09:15:00 | 370.10 | 358.26 | 354.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 378.95 | 381.50 | 375.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 378.95 | 381.50 | 375.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 377.00 | 380.60 | 376.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 371.45 | 380.60 | 376.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 372.70 | 379.02 | 375.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 370.45 | 379.02 | 375.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 370.95 | 377.40 | 375.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:45:00 | 371.60 | 377.40 | 375.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 364.55 | 372.76 | 373.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 10:15:00 | 361.50 | 368.78 | 371.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 392.70 | 370.04 | 370.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 392.70 | 370.04 | 370.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 392.70 | 370.04 | 370.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 395.75 | 370.04 | 370.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 393.45 | 374.72 | 372.34 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 372.10 | 379.27 | 379.32 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 380.45 | 379.33 | 379.27 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 376.70 | 378.80 | 379.04 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 10:15:00 | 389.00 | 380.84 | 379.95 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 375.40 | 380.32 | 380.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 373.55 | 377.48 | 378.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 379.20 | 377.69 | 378.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 379.20 | 377.69 | 378.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 379.20 | 377.69 | 378.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 375.25 | 377.21 | 378.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 356.49 | 365.59 | 371.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 360.65 | 358.07 | 364.23 | SL hit (close>ema200) qty=0.50 sl=358.07 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 353.00 | 346.41 | 345.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 358.70 | 350.96 | 349.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 15:15:00 | 350.00 | 350.77 | 349.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 351.90 | 351.64 | 349.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 351.90 | 351.64 | 349.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 351.90 | 351.64 | 349.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 345.90 | 350.49 | 349.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 345.90 | 350.49 | 349.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 348.70 | 350.13 | 349.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:00:00 | 349.95 | 350.10 | 349.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 349.85 | 350.12 | 349.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 343.00 | 349.26 | 349.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 343.00 | 349.26 | 349.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 341.20 | 347.64 | 348.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 352.05 | 347.07 | 347.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 352.05 | 347.07 | 347.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 352.05 | 347.07 | 347.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 352.05 | 347.07 | 347.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 344.45 | 346.54 | 347.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 343.70 | 346.54 | 347.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:45:00 | 342.50 | 345.05 | 346.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:45:00 | 343.70 | 342.13 | 344.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 343.40 | 342.98 | 344.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 346.05 | 343.59 | 344.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 345.95 | 343.59 | 344.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 343.70 | 343.99 | 344.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 343.70 | 343.99 | 344.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 343.70 | 343.93 | 344.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 342.85 | 343.93 | 344.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 342.30 | 343.61 | 344.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 336.70 | 342.69 | 343.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 337.85 | 340.94 | 342.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:15:00 | 337.25 | 340.53 | 342.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 14:15:00 | 326.51 | 330.96 | 334.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 14:15:00 | 326.51 | 330.96 | 334.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 14:15:00 | 326.23 | 330.96 | 334.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 325.38 | 328.47 | 332.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 319.86 | 328.47 | 332.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 320.96 | 328.47 | 332.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 320.39 | 328.47 | 332.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 316.50 | 316.16 | 319.21 | SL hit (close>ema200) qty=0.50 sl=316.16 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 322.05 | 316.22 | 315.85 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 309.75 | 315.00 | 315.44 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 321.65 | 316.04 | 315.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 324.75 | 320.06 | 318.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 323.05 | 323.21 | 321.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:45:00 | 322.75 | 323.21 | 321.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 322.45 | 322.99 | 321.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 321.65 | 322.99 | 321.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 322.55 | 322.85 | 322.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 322.05 | 322.85 | 322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 322.00 | 322.68 | 322.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 323.35 | 321.97 | 321.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 15:15:00 | 320.00 | 323.15 | 323.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 15:15:00 | 320.00 | 323.15 | 323.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 314.70 | 321.46 | 322.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 310.25 | 309.10 | 310.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:30:00 | 310.00 | 309.10 | 310.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 310.95 | 309.64 | 310.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 310.95 | 309.64 | 310.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 311.70 | 310.06 | 310.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 311.70 | 310.06 | 310.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 310.95 | 310.23 | 310.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 312.00 | 310.23 | 310.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 310.15 | 310.22 | 310.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:45:00 | 311.55 | 310.22 | 310.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 310.75 | 310.32 | 310.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 309.95 | 310.32 | 310.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 311.00 | 310.46 | 310.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 311.00 | 310.46 | 310.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 310.10 | 310.39 | 310.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 309.85 | 310.39 | 310.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 310.25 | 310.36 | 310.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:00:00 | 308.75 | 310.04 | 310.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 308.60 | 309.87 | 310.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 308.25 | 309.54 | 310.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 309.25 | 309.49 | 310.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 310.55 | 309.70 | 310.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:45:00 | 310.60 | 309.70 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 307.75 | 309.31 | 309.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:30:00 | 308.60 | 309.31 | 309.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 308.00 | 309.05 | 309.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 308.00 | 309.05 | 309.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 307.50 | 308.74 | 309.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 302.65 | 308.46 | 309.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 306.75 | 308.53 | 309.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:45:00 | 306.90 | 308.28 | 309.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 305.00 | 307.82 | 308.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 310.30 | 306.85 | 307.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 310.30 | 306.85 | 307.59 | SL hit (close>static) qty=1.00 sl=309.90 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 11:15:00 | 309.75 | 308.17 | 308.11 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 13:15:00 | 306.15 | 308.22 | 308.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 14:15:00 | 303.85 | 307.35 | 308.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 304.50 | 303.56 | 305.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:30:00 | 303.05 | 303.56 | 305.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 303.40 | 303.53 | 304.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 302.45 | 303.32 | 304.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 297.70 | 294.75 | 294.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 297.70 | 294.75 | 294.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 299.50 | 295.70 | 294.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 296.75 | 297.49 | 296.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 296.75 | 297.49 | 296.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 296.75 | 297.49 | 296.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 296.75 | 297.49 | 296.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 297.00 | 297.39 | 296.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 296.20 | 297.39 | 296.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 296.35 | 297.18 | 296.52 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 290.85 | 295.92 | 296.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 288.45 | 292.97 | 294.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 291.15 | 290.93 | 292.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 291.15 | 290.93 | 292.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 294.45 | 291.64 | 293.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 294.05 | 291.64 | 293.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 292.65 | 291.84 | 292.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 294.80 | 291.84 | 292.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 293.90 | 292.25 | 293.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 293.90 | 292.25 | 293.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 293.60 | 292.52 | 293.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 293.60 | 292.52 | 293.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 292.65 | 292.78 | 293.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 289.60 | 292.11 | 292.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:45:00 | 289.30 | 291.33 | 292.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 288.95 | 291.05 | 291.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 289.60 | 290.83 | 291.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 275.12 | 282.03 | 285.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 274.83 | 282.03 | 285.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 274.50 | 282.03 | 285.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 275.12 | 282.03 | 285.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 275.10 | 275.05 | 279.47 | SL hit (close>ema200) qty=0.50 sl=275.05 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 287.50 | 279.09 | 278.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 291.05 | 284.65 | 281.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 289.75 | 289.96 | 286.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:00:00 | 289.75 | 289.96 | 286.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 289.05 | 291.26 | 289.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 289.05 | 291.26 | 289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 290.40 | 291.09 | 289.86 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 282.85 | 288.63 | 289.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 280.10 | 286.92 | 288.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 281.05 | 280.90 | 284.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 281.05 | 280.90 | 284.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 284.25 | 281.57 | 284.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 284.25 | 281.57 | 284.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 284.50 | 282.15 | 284.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 283.20 | 284.37 | 284.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:15:00 | 282.65 | 284.31 | 284.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 269.04 | 274.13 | 278.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 268.52 | 274.13 | 278.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 276.80 | 274.20 | 277.36 | SL hit (close>ema200) qty=0.50 sl=274.20 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 286.50 | 279.84 | 278.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 286.95 | 282.38 | 280.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 324.50 | 327.05 | 324.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 13:15:00 | 324.50 | 327.05 | 324.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 324.50 | 327.05 | 324.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 324.50 | 327.05 | 324.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 326.15 | 326.87 | 324.68 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 322.00 | 323.70 | 323.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 319.40 | 322.17 | 323.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 13:15:00 | 317.50 | 316.37 | 319.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 13:15:00 | 317.50 | 316.37 | 319.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 317.50 | 316.37 | 319.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 317.50 | 316.37 | 319.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 318.45 | 316.79 | 319.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 318.45 | 316.79 | 319.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 304.80 | 314.64 | 317.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 302.15 | 310.43 | 314.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 303.50 | 307.50 | 310.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:45:00 | 303.05 | 306.50 | 309.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 301.85 | 301.09 | 304.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 302.30 | 301.41 | 303.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 302.30 | 301.41 | 303.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 301.20 | 301.61 | 303.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 303.10 | 301.03 | 300.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 303.10 | 301.03 | 300.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 15:15:00 | 304.30 | 301.68 | 301.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 09:15:00 | 301.00 | 301.55 | 301.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 301.00 | 301.55 | 301.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 301.00 | 301.55 | 301.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 300.50 | 301.55 | 301.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 300.35 | 301.31 | 301.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 300.35 | 301.31 | 301.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 300.10 | 301.07 | 301.07 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 10:15:00 | 303.00 | 301.12 | 301.01 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 13:15:00 | 300.60 | 300.96 | 300.97 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 14:15:00 | 305.30 | 301.83 | 301.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 15:15:00 | 308.40 | 303.14 | 302.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 09:15:00 | 298.80 | 302.27 | 301.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 298.80 | 302.27 | 301.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 298.80 | 302.27 | 301.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:00:00 | 298.80 | 302.27 | 301.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 300.20 | 301.86 | 301.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 13:00:00 | 303.50 | 302.03 | 301.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 14:00:00 | 303.20 | 304.71 | 303.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 303.40 | 304.45 | 303.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 302.00 | 303.58 | 303.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 302.00 | 303.58 | 303.58 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 304.45 | 303.71 | 303.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 305.95 | 304.59 | 304.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 304.60 | 304.71 | 304.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 304.60 | 304.71 | 304.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 304.60 | 304.71 | 304.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:30:00 | 305.25 | 304.71 | 304.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 304.90 | 304.81 | 304.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 304.80 | 304.81 | 304.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 304.00 | 304.65 | 304.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 303.10 | 304.65 | 304.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 302.85 | 304.29 | 304.24 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 303.00 | 304.11 | 304.23 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 10:15:00 | 304.85 | 304.37 | 304.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 12:15:00 | 305.05 | 304.60 | 304.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 304.35 | 304.61 | 304.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 14:15:00 | 304.35 | 304.61 | 304.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 304.35 | 304.61 | 304.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 305.00 | 304.61 | 304.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 304.00 | 304.49 | 304.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 301.40 | 304.49 | 304.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 300.70 | 303.73 | 304.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 299.80 | 302.58 | 303.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 304.70 | 295.10 | 296.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 304.70 | 295.10 | 296.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 304.70 | 295.10 | 296.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 305.65 | 295.10 | 296.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 303.00 | 296.68 | 297.26 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 303.85 | 298.12 | 297.85 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 11:15:00 | 295.85 | 299.00 | 299.41 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 303.80 | 299.53 | 299.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 307.60 | 302.84 | 301.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 305.20 | 305.44 | 303.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 305.20 | 305.44 | 303.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 304.35 | 305.22 | 303.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 307.30 | 305.22 | 303.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 10:15:00 | 305.35 | 305.18 | 303.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 302.00 | 304.07 | 303.41 | SL hit (close<static) qty=1.00 sl=302.20 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 300.70 | 303.09 | 303.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 11:15:00 | 297.45 | 298.96 | 300.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 297.85 | 296.92 | 298.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 297.85 | 296.92 | 298.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 297.85 | 296.92 | 298.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 297.30 | 296.92 | 298.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 295.85 | 296.71 | 298.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 294.80 | 296.89 | 298.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 299.45 | 297.60 | 298.25 | SL hit (close>static) qty=1.00 sl=299.15 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 301.05 | 298.48 | 298.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 301.95 | 300.15 | 299.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 297.95 | 300.09 | 299.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 297.95 | 300.09 | 299.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 297.95 | 300.09 | 299.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 297.95 | 300.09 | 299.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 298.35 | 299.74 | 299.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 297.50 | 299.74 | 299.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 297.85 | 299.36 | 299.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 285.60 | 296.36 | 297.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 293.70 | 293.43 | 295.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 293.70 | 293.43 | 295.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 293.55 | 293.45 | 295.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 297.05 | 293.45 | 295.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 299.80 | 294.72 | 295.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 299.80 | 294.72 | 295.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 302.25 | 296.23 | 296.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:30:00 | 301.70 | 296.23 | 296.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 302.20 | 297.42 | 296.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 302.85 | 298.51 | 297.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 10:15:00 | 300.80 | 301.81 | 299.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 11:00:00 | 300.80 | 301.81 | 299.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 300.20 | 301.15 | 299.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:30:00 | 299.65 | 301.15 | 299.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 299.70 | 300.86 | 299.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:30:00 | 300.30 | 300.86 | 299.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 300.35 | 300.76 | 299.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 302.00 | 300.76 | 299.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 297.25 | 300.06 | 299.74 | SL hit (close<static) qty=1.00 sl=299.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 10:15:00 | 297.35 | 299.51 | 299.52 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 301.90 | 299.99 | 299.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 303.55 | 300.70 | 300.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 333.00 | 335.75 | 331.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 333.00 | 335.75 | 331.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 334.90 | 335.45 | 333.80 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 325.50 | 331.71 | 332.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 320.20 | 326.57 | 328.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 320.25 | 320.18 | 322.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 319.70 | 320.18 | 322.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 319.40 | 320.03 | 322.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:15:00 | 324.50 | 320.03 | 322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 324.75 | 320.97 | 322.26 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 324.85 | 323.06 | 322.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 327.60 | 323.97 | 323.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 319.55 | 323.09 | 323.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 319.55 | 323.09 | 323.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 319.55 | 323.09 | 323.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 319.55 | 323.09 | 323.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 319.55 | 322.38 | 322.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 318.30 | 321.56 | 322.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 321.20 | 319.85 | 321.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 321.20 | 319.85 | 321.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 321.20 | 319.85 | 321.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 321.60 | 319.85 | 321.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 324.60 | 320.80 | 321.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:45:00 | 323.00 | 320.80 | 321.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 320.50 | 320.74 | 321.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:00:00 | 319.05 | 320.40 | 321.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 318.55 | 320.25 | 320.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 324.80 | 321.36 | 321.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 324.80 | 321.36 | 321.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 329.85 | 323.06 | 322.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 316.50 | 322.97 | 322.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 316.50 | 322.97 | 322.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 316.50 | 322.97 | 322.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 316.50 | 322.97 | 322.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 315.45 | 321.47 | 321.97 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 323.05 | 320.63 | 320.51 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 14:15:00 | 320.45 | 320.79 | 320.82 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 321.20 | 320.81 | 320.81 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 320.50 | 320.75 | 320.78 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 322.20 | 320.97 | 320.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 325.05 | 322.00 | 321.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 327.25 | 325.53 | 324.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:30:00 | 325.85 | 328.88 | 328.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 334.00 | 339.94 | 340.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 334.00 | 339.94 | 340.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 309.30 | 331.30 | 334.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 310.45 | 308.08 | 311.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 310.45 | 308.08 | 311.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 311.70 | 308.81 | 311.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 311.70 | 308.81 | 311.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 312.85 | 309.62 | 311.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 312.85 | 309.62 | 311.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 314.35 | 310.56 | 312.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:45:00 | 314.40 | 310.56 | 312.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 316.00 | 313.52 | 313.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 322.30 | 315.27 | 314.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 328.15 | 328.99 | 325.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 329.45 | 328.99 | 325.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 321.85 | 327.03 | 325.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 321.85 | 327.03 | 325.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 324.50 | 326.53 | 325.38 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 321.30 | 324.48 | 324.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 320.15 | 323.61 | 324.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 320.00 | 319.22 | 320.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 320.00 | 319.22 | 320.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 316.50 | 318.78 | 320.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 312.30 | 316.47 | 317.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 312.00 | 313.97 | 314.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 318.80 | 314.81 | 314.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 318.80 | 314.81 | 314.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 320.55 | 315.95 | 315.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 322.50 | 322.81 | 320.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:45:00 | 322.30 | 322.81 | 320.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 320.25 | 322.33 | 320.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 320.25 | 322.33 | 320.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 317.30 | 321.32 | 320.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 317.30 | 321.32 | 320.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 320.15 | 320.24 | 320.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:15:00 | 323.00 | 320.24 | 320.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 319.20 | 320.47 | 320.25 | SL hit (close<static) qty=1.00 sl=319.80 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 318.50 | 319.91 | 320.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 13:15:00 | 317.75 | 319.27 | 319.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:15:00 | 323.75 | 319.76 | 319.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 324.05 | 320.62 | 320.18 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 320.00 | 321.23 | 321.25 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 322.40 | 321.46 | 321.35 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 321.00 | 321.76 | 321.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 320.40 | 321.49 | 321.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 322.95 | 320.63 | 321.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 322.10 | 320.93 | 321.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 322.25 | 320.93 | 321.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 320.70 | 321.12 | 321.24 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 324.00 | 321.69 | 321.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 325.65 | 322.48 | 321.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 11:15:00 | 346.20 | 347.03 | 344.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:30:00 | 345.65 | 347.03 | 344.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 342.30 | 345.89 | 344.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 341.80 | 345.89 | 344.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 345.05 | 345.72 | 344.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 346.60 | 345.35 | 344.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 346.30 | 345.27 | 344.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:00:00 | 346.25 | 345.49 | 344.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:00:00 | 346.35 | 345.74 | 345.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 346.00 | 345.79 | 345.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 345.90 | 345.79 | 345.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 335.60 | 343.76 | 344.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 335.60 | 343.76 | 344.39 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 348.65 | 345.11 | 344.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 350.60 | 346.48 | 345.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 349.85 | 350.24 | 348.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 349.05 | 350.00 | 348.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 348.75 | 350.00 | 348.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 346.30 | 349.26 | 348.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 346.30 | 349.26 | 348.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 347.75 | 348.96 | 348.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:15:00 | 349.00 | 348.96 | 348.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 348.80 | 348.93 | 348.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 349.80 | 348.62 | 348.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:00:00 | 350.05 | 349.85 | 349.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 349.95 | 349.55 | 349.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 353.85 | 350.51 | 349.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 349.95 | 351.08 | 350.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 349.50 | 351.08 | 350.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 347.50 | 350.36 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 350.00 | 350.36 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 347.50 | 349.79 | 349.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 347.50 | 349.79 | 349.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 346.90 | 349.21 | 349.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 346.90 | 349.21 | 349.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 344.45 | 347.14 | 348.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 10:15:00 | 333.90 | 332.60 | 336.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 11:00:00 | 333.90 | 332.60 | 336.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 335.80 | 333.24 | 336.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 336.25 | 333.24 | 336.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 334.85 | 333.56 | 336.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 331.55 | 333.56 | 335.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 338.00 | 335.06 | 335.80 | SL hit (close>static) qty=1.00 sl=337.45 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 13:15:00 | 337.55 | 335.26 | 335.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 09:15:00 | 342.15 | 337.26 | 336.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 344.75 | 346.33 | 342.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 342.05 | 345.15 | 343.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 342.05 | 345.15 | 343.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 343.90 | 344.90 | 343.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:45:00 | 344.40 | 343.89 | 343.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 345.75 | 344.01 | 343.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 340.30 | 343.15 | 343.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 340.30 | 343.15 | 343.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 334.15 | 338.54 | 340.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 337.75 | 335.14 | 336.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 337.45 | 335.60 | 336.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 337.45 | 335.60 | 336.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 340.95 | 336.67 | 337.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 340.95 | 336.67 | 337.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 340.50 | 337.44 | 337.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 340.50 | 337.44 | 337.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 344.85 | 338.92 | 338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 345.95 | 341.65 | 339.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 358.70 | 359.06 | 356.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:30:00 | 359.10 | 359.06 | 356.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 355.35 | 358.32 | 356.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 355.35 | 358.32 | 356.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 355.00 | 357.66 | 355.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 354.85 | 357.66 | 355.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 356.50 | 356.84 | 355.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 356.15 | 356.84 | 355.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 355.50 | 356.57 | 355.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 355.50 | 356.57 | 355.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 357.35 | 356.73 | 355.96 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 355.45 | 355.75 | 355.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 352.30 | 354.82 | 355.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 319.50 | 319.45 | 323.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:30:00 | 318.50 | 319.45 | 323.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 323.85 | 319.45 | 321.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 322.40 | 319.45 | 321.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 326.00 | 320.76 | 321.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 328.50 | 320.76 | 321.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 327.00 | 322.66 | 322.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 332.45 | 325.49 | 323.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 328.00 | 331.28 | 328.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 330.15 | 331.06 | 328.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 333.50 | 331.06 | 328.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:45:00 | 331.95 | 331.79 | 329.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 339.30 | 340.72 | 340.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 339.30 | 340.72 | 340.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 338.35 | 339.99 | 340.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 338.05 | 334.73 | 336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 335.40 | 334.86 | 336.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 335.40 | 334.86 | 336.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 334.00 | 334.69 | 335.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 330.75 | 334.10 | 334.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 332.10 | 330.71 | 331.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 329.95 | 331.50 | 331.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 315.50 | 319.44 | 321.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 314.21 | 317.90 | 320.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 313.45 | 317.90 | 320.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 313.55 | 313.32 | 316.45 | SL hit (close>ema200) qty=0.50 sl=313.32 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 320.40 | 316.72 | 316.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 323.00 | 320.61 | 319.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 326.45 | 326.47 | 323.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:30:00 | 326.95 | 326.47 | 323.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 325.10 | 326.13 | 323.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 324.75 | 326.13 | 323.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 324.20 | 325.74 | 324.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 324.40 | 325.74 | 324.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 322.55 | 325.10 | 323.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 322.55 | 325.10 | 323.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 323.00 | 324.68 | 323.80 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 320.25 | 322.99 | 323.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 318.60 | 322.12 | 322.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 319.85 | 313.89 | 313.76 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 313.45 | 314.12 | 314.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 311.00 | 313.50 | 313.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 313.30 | 311.25 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 313.25 | 311.65 | 312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 312.80 | 311.65 | 312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 312.45 | 311.81 | 312.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 314.00 | 311.81 | 312.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 313.10 | 312.07 | 312.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 313.10 | 312.07 | 312.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 312.35 | 312.13 | 312.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:00:00 | 310.85 | 311.87 | 312.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 310.70 | 308.03 | 309.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 312.15 | 309.42 | 309.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 312.15 | 309.42 | 309.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 314.50 | 310.82 | 309.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 310.40 | 313.07 | 311.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 311.35 | 312.73 | 311.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 309.90 | 312.73 | 311.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 309.40 | 312.06 | 311.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 309.40 | 312.06 | 311.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 310.40 | 310.84 | 310.86 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 316.25 | 311.72 | 311.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 12:15:00 | 319.70 | 315.03 | 313.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 315.00 | 317.18 | 315.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 314.40 | 316.62 | 315.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 314.40 | 316.62 | 315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 312.95 | 315.89 | 314.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 312.95 | 315.89 | 314.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 314.85 | 314.84 | 314.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:15:00 | 313.10 | 314.84 | 314.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 311.80 | 314.23 | 314.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 309.55 | 312.01 | 313.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 314.35 | 311.48 | 312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 317.25 | 312.63 | 312.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 317.25 | 312.63 | 312.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 316.45 | 313.39 | 313.12 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 311.75 | 312.80 | 312.91 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 313.50 | 312.88 | 312.86 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 307.25 | 313.05 | 313.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 12:15:00 | 302.80 | 311.00 | 312.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 294.05 | 293.93 | 298.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 294.05 | 293.93 | 298.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 287.40 | 285.05 | 286.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:30:00 | 288.50 | 285.05 | 286.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 287.50 | 285.54 | 286.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 286.30 | 285.54 | 286.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 288.80 | 286.56 | 286.73 | SL hit (close>static) qty=1.00 sl=287.90 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 290.40 | 287.33 | 287.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 293.30 | 290.81 | 289.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 11:15:00 | 291.55 | 292.61 | 291.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 11:45:00 | 291.80 | 292.61 | 291.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 290.90 | 292.26 | 291.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 290.40 | 292.26 | 291.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 291.30 | 292.07 | 291.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:30:00 | 290.95 | 292.07 | 291.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 288.40 | 291.34 | 290.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 288.40 | 291.34 | 290.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 288.50 | 290.77 | 290.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 287.55 | 290.77 | 290.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 287.80 | 290.18 | 290.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 284.55 | 287.76 | 288.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 279.70 | 279.45 | 282.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 11:00:00 | 279.70 | 279.45 | 282.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 285.30 | 280.52 | 281.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 285.30 | 280.52 | 281.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 286.00 | 281.62 | 282.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 283.70 | 281.62 | 282.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 285.85 | 282.65 | 282.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 285.85 | 282.65 | 282.29 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 281.45 | 282.07 | 282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 279.60 | 281.57 | 281.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 281.95 | 281.57 | 281.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 280.85 | 281.42 | 281.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 280.85 | 281.42 | 281.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 277.60 | 280.56 | 281.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 276.90 | 280.56 | 281.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 283.60 | 280.66 | 280.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 283.60 | 280.66 | 280.53 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 278.90 | 280.45 | 280.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 276.75 | 279.53 | 280.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 276.20 | 275.77 | 277.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 276.20 | 275.77 | 277.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 277.70 | 276.15 | 277.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 278.70 | 276.72 | 277.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 281.05 | 277.59 | 277.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 281.30 | 277.59 | 277.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 281.25 | 278.32 | 278.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 283.55 | 279.80 | 278.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 280.40 | 281.52 | 280.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 280.10 | 281.24 | 280.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:00:00 | 281.95 | 281.21 | 280.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 281.40 | 281.26 | 280.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 282.85 | 281.01 | 280.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 278.45 | 280.50 | 280.21 | SL hit (close<static) qty=1.00 sl=279.35 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 279.45 | 279.99 | 280.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 278.80 | 279.75 | 279.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:15:00 | 283.35 | 279.33 | 279.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 284.95 | 280.45 | 280.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 287.00 | 281.76 | 280.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 12:15:00 | 292.90 | 293.12 | 290.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 13:00:00 | 292.90 | 293.12 | 290.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 293.35 | 293.17 | 290.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:45:00 | 291.45 | 293.17 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 291.00 | 292.46 | 291.18 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 289.55 | 290.49 | 290.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 288.85 | 290.24 | 290.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 289.25 | 287.14 | 288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 288.20 | 287.36 | 288.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 288.20 | 287.36 | 288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 283.25 | 283.42 | 284.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 283.25 | 283.42 | 284.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 282.95 | 283.28 | 284.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 281.80 | 282.82 | 283.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:30:00 | 282.00 | 282.48 | 283.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 282.05 | 282.48 | 283.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 282.05 | 282.42 | 283.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 281.95 | 282.29 | 282.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 280.75 | 281.96 | 282.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 277.25 | 281.82 | 282.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 278.05 | 277.04 | 279.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 286.95 | 280.60 | 279.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 286.95 | 280.60 | 279.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 289.00 | 282.28 | 280.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 278.90 | 281.60 | 280.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 277.95 | 280.87 | 280.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 277.95 | 280.87 | 280.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 278.50 | 279.78 | 279.86 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 282.35 | 280.22 | 280.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 285.00 | 283.01 | 281.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 283.05 | 286.64 | 285.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 285.90 | 286.49 | 285.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 287.35 | 286.43 | 285.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 282.90 | 285.61 | 285.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 282.90 | 285.61 | 285.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 279.55 | 283.66 | 284.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 275.60 | 275.18 | 277.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:00:00 | 274.15 | 274.98 | 277.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:30:00 | 273.85 | 274.69 | 276.81 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 12:00:00 | 273.55 | 274.69 | 276.81 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 14:00:00 | 273.60 | 274.44 | 276.33 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 273.45 | 274.20 | 275.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 275.50 | 274.20 | 275.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 266.00 | 267.14 | 269.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 265.70 | 266.29 | 268.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 260.44 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 260.16 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 259.87 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 259.92 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 261.40 | 261.24 | 263.70 | SL hit (close>ema200) qty=0.50 sl=261.24 alert=retest1 |

### Cycle 127 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 266.90 | 264.36 | 264.23 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 262.40 | 264.10 | 264.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 261.20 | 263.52 | 263.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 263.45 | 262.47 | 263.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 260.95 | 262.17 | 263.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 262.50 | 262.17 | 263.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 265.00 | 262.73 | 263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:45:00 | 265.40 | 262.73 | 263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 264.85 | 263.16 | 263.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 264.90 | 263.16 | 263.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 266.90 | 264.16 | 263.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 268.95 | 265.12 | 264.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 269.70 | 270.00 | 267.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:30:00 | 269.45 | 270.00 | 267.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 273.05 | 270.57 | 269.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 273.95 | 271.25 | 269.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:00:00 | 274.35 | 272.22 | 270.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 266.90 | 270.31 | 270.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 266.90 | 270.31 | 270.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 261.35 | 268.52 | 269.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 266.75 | 264.95 | 267.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 266.75 | 264.95 | 267.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 266.75 | 265.31 | 267.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 276.00 | 265.31 | 267.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 278.15 | 267.88 | 268.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:15:00 | 279.30 | 267.88 | 268.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 280.15 | 270.33 | 269.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 280.95 | 275.00 | 271.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 278.50 | 280.13 | 276.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 276.80 | 279.47 | 276.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 270.50 | 279.47 | 276.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 272.45 | 278.06 | 276.49 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 270.20 | 275.50 | 275.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 268.95 | 273.48 | 274.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 262.85 | 262.48 | 266.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 262.65 | 262.48 | 266.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 266.05 | 264.05 | 266.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 266.05 | 264.05 | 266.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 266.55 | 264.55 | 266.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 265.25 | 264.55 | 266.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 267.20 | 265.08 | 266.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 268.20 | 265.08 | 266.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 260.40 | 264.15 | 265.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 258.80 | 261.18 | 263.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 245.86 | 253.22 | 256.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 250.80 | 250.64 | 253.60 | SL hit (close>ema200) qty=0.50 sl=250.64 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 247.35 | 243.03 | 242.53 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 241.80 | 244.21 | 244.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 231.43 | 238.50 | 241.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 230.99 | 229.95 | 232.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 227.50 | 229.84 | 232.17 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 228.16 | 229.56 | 231.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 219.97 | 229.37 | 231.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 226.95 | 223.74 | 226.28 | SL hit (close>ema400) qty=1.00 sl=226.28 alert=retest1 |

### Cycle 135 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 231.62 | 228.21 | 227.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 233.52 | 229.96 | 228.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 229.66 | 229.91 | 228.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:30:00 | 230.00 | 229.91 | 228.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 228.37 | 229.58 | 229.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 228.37 | 229.58 | 229.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 228.43 | 229.35 | 228.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 225.64 | 229.35 | 228.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 226.70 | 228.82 | 228.74 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 227.97 | 228.65 | 228.67 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 229.64 | 228.85 | 228.76 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 227.03 | 228.38 | 228.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 226.21 | 227.88 | 228.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 223.16 | 221.70 | 223.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 224.23 | 222.26 | 223.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 224.23 | 222.26 | 223.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 223.18 | 222.44 | 223.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:00:00 | 221.57 | 222.27 | 223.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:00:00 | 222.02 | 222.11 | 223.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:45:00 | 222.10 | 221.89 | 222.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.49 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.92 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.99 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 199.41 | 208.04 | 211.90 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 207.06 | 204.96 | 204.91 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 202.93 | 204.82 | 204.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 199.69 | 203.79 | 204.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 200.89 | 199.69 | 201.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:00:00 | 200.89 | 199.69 | 201.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 204.89 | 198.75 | 200.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 204.89 | 198.75 | 200.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 205.22 | 200.04 | 200.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 205.22 | 200.04 | 200.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 204.83 | 201.77 | 201.44 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 196.25 | 201.52 | 201.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 12:15:00 | 195.29 | 197.22 | 198.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 197.46 | 197.37 | 198.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 199.85 | 197.87 | 198.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 199.85 | 197.87 | 198.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 200.00 | 198.30 | 198.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 205.92 | 198.30 | 198.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 208.68 | 200.37 | 199.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 211.80 | 205.60 | 202.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 245.10 | 246.64 | 241.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 245.10 | 246.64 | 241.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 244.83 | 246.28 | 241.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 241.69 | 246.28 | 241.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 244.60 | 245.43 | 243.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 251.55 | 245.43 | 243.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 256.35 | 258.20 | 258.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 256.35 | 258.20 | 258.42 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 263.30 | 259.41 | 258.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 263.70 | 260.27 | 259.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:30:00 | 265.50 | 261.78 | 261.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 265.50 | 262.61 | 261.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 265.45 | 262.96 | 261.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 257.65 | 261.32 | 261.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 257.65 | 261.32 | 261.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 256.85 | 260.42 | 261.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 265.15 | 261.33 | 260.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 266.15 | 262.77 | 261.72 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-19 10:00:00 | 349.20 | 2024-06-21 09:15:00 | 339.45 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-06-19 11:45:00 | 349.55 | 2024-06-21 09:15:00 | 339.45 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-06-19 13:15:00 | 348.80 | 2024-06-21 09:15:00 | 339.45 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-06-25 10:15:00 | 335.35 | 2024-07-01 09:15:00 | 341.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-07-03 09:15:00 | 344.35 | 2024-07-05 14:15:00 | 339.85 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-07-11 12:15:00 | 331.65 | 2024-07-22 12:15:00 | 325.00 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2024-07-31 11:45:00 | 315.90 | 2024-08-14 14:15:00 | 300.77 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2024-07-31 12:30:00 | 315.60 | 2024-08-14 15:15:00 | 300.10 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2024-07-31 13:45:00 | 316.00 | 2024-08-14 15:15:00 | 299.82 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2024-07-31 14:30:00 | 315.75 | 2024-08-14 15:15:00 | 300.20 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2024-08-01 12:15:00 | 316.60 | 2024-08-14 15:15:00 | 299.96 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2024-08-01 13:30:00 | 315.60 | 2024-08-14 15:15:00 | 299.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 11:45:00 | 315.90 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2024-07-31 12:30:00 | 315.60 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2024-07-31 13:45:00 | 316.00 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2024-07-31 14:30:00 | 315.75 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2024-08-01 12:15:00 | 316.60 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2024-08-01 13:30:00 | 315.60 | 2024-08-16 13:15:00 | 303.05 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2024-08-05 09:15:00 | 312.45 | 2024-08-19 14:15:00 | 304.50 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2024-08-30 13:30:00 | 322.80 | 2024-09-04 10:15:00 | 319.80 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-08-30 15:00:00 | 331.35 | 2024-09-04 10:15:00 | 319.80 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2024-09-04 09:30:00 | 326.80 | 2024-09-04 10:15:00 | 319.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-09-10 11:30:00 | 322.60 | 2024-09-11 10:15:00 | 324.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-10 15:15:00 | 322.25 | 2024-09-11 10:15:00 | 324.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-09-25 09:15:00 | 367.50 | 2024-09-26 15:15:00 | 360.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-09-25 12:30:00 | 364.25 | 2024-09-26 15:15:00 | 360.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-21 10:30:00 | 375.25 | 2024-10-22 10:15:00 | 356.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:30:00 | 375.25 | 2024-10-23 09:15:00 | 360.65 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2024-11-04 12:00:00 | 349.95 | 2024-11-05 09:15:00 | 343.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-04 13:45:00 | 349.85 | 2024-11-05 09:15:00 | 343.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-11-05 15:15:00 | 343.70 | 2024-11-12 14:15:00 | 326.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 09:45:00 | 342.50 | 2024-11-12 14:15:00 | 326.51 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2024-11-07 09:45:00 | 343.70 | 2024-11-12 14:15:00 | 326.23 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-11-07 10:30:00 | 343.40 | 2024-11-13 09:15:00 | 325.38 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2024-11-08 11:15:00 | 336.70 | 2024-11-13 09:15:00 | 319.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:15:00 | 337.85 | 2024-11-13 09:15:00 | 320.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 15:15:00 | 337.25 | 2024-11-13 09:15:00 | 320.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-05 15:15:00 | 343.70 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 7.91% |
| SELL | retest2 | 2024-11-06 09:45:00 | 342.50 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 7.59% |
| SELL | retest2 | 2024-11-07 09:45:00 | 343.70 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 7.91% |
| SELL | retest2 | 2024-11-07 10:30:00 | 343.40 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 7.83% |
| SELL | retest2 | 2024-11-08 11:15:00 | 336.70 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest2 | 2024-11-08 14:15:00 | 337.85 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 6.32% |
| SELL | retest2 | 2024-11-08 15:15:00 | 337.25 | 2024-11-18 13:15:00 | 316.50 | STOP_HIT | 0.50 | 6.15% |
| BUY | retest2 | 2024-12-06 09:15:00 | 323.35 | 2024-12-09 15:15:00 | 320.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-12-17 13:00:00 | 308.75 | 2024-12-23 09:15:00 | 310.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-12-18 09:15:00 | 308.60 | 2024-12-23 09:15:00 | 310.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-12-18 10:00:00 | 308.25 | 2024-12-23 09:15:00 | 310.30 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-18 11:00:00 | 309.25 | 2024-12-23 09:15:00 | 310.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-12-19 09:15:00 | 302.65 | 2024-12-23 10:15:00 | 311.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-12-19 10:15:00 | 306.75 | 2024-12-23 10:15:00 | 311.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-19 10:45:00 | 306.90 | 2024-12-23 10:15:00 | 311.50 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-12-20 13:00:00 | 305.00 | 2024-12-23 10:15:00 | 311.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-12-27 12:45:00 | 302.45 | 2025-01-02 13:15:00 | 297.70 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-01-08 12:00:00 | 289.60 | 2025-01-13 09:15:00 | 275.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:45:00 | 289.30 | 2025-01-13 09:15:00 | 274.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 288.95 | 2025-01-13 09:15:00 | 274.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 289.60 | 2025-01-13 09:15:00 | 275.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:00:00 | 289.60 | 2025-01-14 09:15:00 | 275.10 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2025-01-08 13:45:00 | 289.30 | 2025-01-14 09:15:00 | 275.10 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-01-09 09:15:00 | 288.95 | 2025-01-14 09:15:00 | 275.10 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-01-09 10:15:00 | 289.60 | 2025-01-14 09:15:00 | 275.10 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2025-01-15 10:15:00 | 273.35 | 2025-01-15 13:15:00 | 282.80 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-01-15 11:45:00 | 274.10 | 2025-01-15 13:15:00 | 282.80 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-01-24 13:15:00 | 283.20 | 2025-01-28 09:15:00 | 269.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:15:00 | 282.65 | 2025-01-28 09:15:00 | 268.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 283.20 | 2025-01-28 11:15:00 | 276.80 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2025-01-24 14:15:00 | 282.65 | 2025-01-28 11:15:00 | 276.80 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-02-13 09:15:00 | 302.15 | 2025-02-25 14:15:00 | 303.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-02-14 09:15:00 | 303.50 | 2025-02-25 14:15:00 | 303.10 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-02-14 09:45:00 | 303.05 | 2025-02-25 14:15:00 | 303.10 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-02-17 10:45:00 | 301.85 | 2025-02-25 14:15:00 | 303.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-03-03 13:00:00 | 303.50 | 2025-03-05 09:15:00 | 302.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-03-04 14:00:00 | 303.20 | 2025-03-05 09:15:00 | 302.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-03-04 15:00:00 | 303.40 | 2025-03-05 09:15:00 | 302.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-03-25 09:15:00 | 307.30 | 2025-03-25 12:15:00 | 302.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-03-25 10:15:00 | 305.35 | 2025-03-25 12:15:00 | 302.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-04-01 11:30:00 | 294.80 | 2025-04-01 13:15:00 | 299.45 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-04-02 09:15:00 | 294.30 | 2025-04-02 12:15:00 | 301.05 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-04-11 09:15:00 | 302.00 | 2025-04-11 09:15:00 | 297.25 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-07 13:00:00 | 319.05 | 2025-05-08 10:15:00 | 324.80 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-05-08 09:15:00 | 318.55 | 2025-05-08 10:15:00 | 324.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-16 15:00:00 | 327.25 | 2025-05-29 09:15:00 | 334.00 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2025-05-20 14:30:00 | 325.85 | 2025-05-29 09:15:00 | 334.00 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-06-19 12:15:00 | 312.30 | 2025-06-24 09:15:00 | 318.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-23 09:15:00 | 312.00 | 2025-06-24 09:15:00 | 318.80 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-06-26 15:15:00 | 323.00 | 2025-06-27 09:15:00 | 319.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-15 09:15:00 | 346.60 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-07-15 13:00:00 | 346.30 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-16 12:00:00 | 346.25 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-07-16 14:00:00 | 346.35 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-07-23 10:30:00 | 349.80 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-24 10:00:00 | 350.05 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-24 13:00:00 | 349.95 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-24 15:00:00 | 353.85 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-31 10:00:00 | 331.55 | 2025-07-31 12:15:00 | 338.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-07-31 13:45:00 | 331.70 | 2025-08-01 09:15:00 | 342.75 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-08-06 12:45:00 | 344.40 | 2025-08-07 13:15:00 | 340.30 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-06 15:00:00 | 345.75 | 2025-08-07 13:15:00 | 340.30 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-05 09:15:00 | 333.50 | 2025-09-11 12:15:00 | 339.30 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2025-09-05 11:45:00 | 331.95 | 2025-09-11 12:15:00 | 339.30 | STOP_HIT | 1.00 | 2.21% |
| SELL | retest2 | 2025-09-17 11:15:00 | 330.75 | 2025-09-25 14:15:00 | 315.50 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-09-19 10:30:00 | 332.10 | 2025-09-26 09:15:00 | 314.21 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2025-09-19 15:15:00 | 329.95 | 2025-09-26 09:15:00 | 313.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:15:00 | 330.75 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2025-09-19 10:30:00 | 332.10 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2025-09-19 15:15:00 | 329.95 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2025-10-15 13:00:00 | 310.85 | 2025-10-20 12:15:00 | 312.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-17 11:45:00 | 310.70 | 2025-10-20 12:15:00 | 312.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-14 09:15:00 | 286.30 | 2025-11-14 10:15:00 | 288.80 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-25 09:15:00 | 283.70 | 2025-11-26 11:15:00 | 285.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 10:15:00 | 276.90 | 2025-12-01 14:15:00 | 283.60 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-12-05 14:00:00 | 281.95 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-05 14:45:00 | 281.40 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-08 09:15:00 | 282.85 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-08 11:45:00 | 281.20 | 2025-12-08 12:15:00 | 279.45 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-22 15:15:00 | 281.80 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-23 11:30:00 | 282.00 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-23 12:00:00 | 282.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-23 13:30:00 | 282.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-26 15:00:00 | 280.75 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-29 09:15:00 | 277.25 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-12-29 15:00:00 | 278.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-01-07 13:45:00 | 287.35 | 2026-01-08 11:15:00 | 282.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2026-01-13 11:00:00 | 274.15 | 2026-01-21 09:15:00 | 260.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 11:30:00 | 273.85 | 2026-01-21 09:15:00 | 260.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 12:00:00 | 273.55 | 2026-01-21 09:15:00 | 259.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 14:00:00 | 273.60 | 2026-01-21 09:15:00 | 259.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 11:00:00 | 274.15 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest1 | 2026-01-13 11:30:00 | 273.85 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest1 | 2026-01-13 12:00:00 | 273.55 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest1 | 2026-01-13 14:00:00 | 273.60 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2026-01-20 13:30:00 | 265.70 | 2026-01-23 09:15:00 | 266.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-30 11:00:00 | 273.95 | 2026-02-01 15:15:00 | 266.90 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-30 14:00:00 | 274.35 | 2026-02-01 15:15:00 | 266.90 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-11 09:30:00 | 258.80 | 2026-02-13 09:15:00 | 245.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 258.80 | 2026-02-13 15:15:00 | 250.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest1 | 2026-03-06 10:30:00 | 227.50 | 2026-03-10 09:15:00 | 226.95 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-03-09 09:15:00 | 219.97 | 2026-03-10 14:15:00 | 231.62 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-03-17 12:00:00 | 221.57 | 2026-03-19 13:15:00 | 210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:00:00 | 222.02 | 2026-03-19 13:15:00 | 210.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 10:45:00 | 222.10 | 2026-03-19 13:15:00 | 210.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 12:00:00 | 221.57 | 2026-03-23 09:15:00 | 199.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 14:00:00 | 222.02 | 2026-03-23 09:15:00 | 199.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 10:45:00 | 222.10 | 2026-03-23 09:15:00 | 199.89 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 09:15:00 | 251.55 | 2026-04-29 09:15:00 | 256.35 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2026-05-04 09:30:00 | 265.50 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-05-04 10:45:00 | 265.50 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-05-04 11:45:00 | 265.45 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.94% |
