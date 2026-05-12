# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 742.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 250 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 81 |
| ALERT3 | 381 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 198 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 198 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 205 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 164
- **Target hits / Stop hits / Partials:** 2 / 198 / 5
- **Avg / median % per leg:** -0.55% / -0.74%
- **Sum % (uncompounded):** -112.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 114 | 23 | 20.2% | 2 | 112 | 0 | -0.51% | -57.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| BUY @ 3rd Alert (retest2) | 113 | 23 | 20.4% | 2 | 111 | 0 | -0.49% | -55.2% |
| SELL (all) | 91 | 18 | 19.8% | 0 | 86 | 5 | -0.60% | -55.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.62% | -1.6% |
| SELL @ 3rd Alert (retest2) | 90 | 18 | 20.0% | 0 | 85 | 5 | -0.59% | -53.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.14% | -4.3% |
| retest2 (combined) | 203 | 41 | 20.2% | 2 | 196 | 5 | -0.54% | -108.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 13:15:00 | 255.35 | 257.86 | 257.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 14:15:00 | 254.40 | 257.16 | 257.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 259.90 | 257.45 | 257.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 259.90 | 257.45 | 257.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 259.90 | 257.45 | 257.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 260.50 | 257.45 | 257.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 258.30 | 257.62 | 257.73 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 259.50 | 257.99 | 257.89 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 11:15:00 | 257.25 | 257.81 | 257.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 12:15:00 | 256.75 | 257.60 | 257.78 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 13:15:00 | 259.85 | 258.05 | 257.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 11:15:00 | 263.00 | 259.99 | 259.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 264.25 | 270.37 | 266.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 264.25 | 270.37 | 266.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 264.25 | 270.37 | 266.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 262.65 | 270.37 | 266.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 265.20 | 269.34 | 266.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:30:00 | 264.05 | 269.34 | 266.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 267.75 | 269.02 | 266.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 13:45:00 | 270.40 | 269.35 | 267.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 10:15:00 | 269.35 | 269.21 | 267.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 11:15:00 | 270.15 | 269.15 | 267.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 11:15:00 | 263.55 | 268.03 | 267.31 | SL hit (close<static) qty=1.00 sl=264.85 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 14:15:00 | 264.50 | 266.55 | 266.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 15:15:00 | 259.45 | 265.13 | 266.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 11:15:00 | 261.45 | 260.90 | 262.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-31 12:00:00 | 261.45 | 260.90 | 262.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 262.50 | 261.22 | 262.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:00:00 | 262.50 | 261.22 | 262.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 268.35 | 262.61 | 262.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 268.35 | 262.61 | 262.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 15:15:00 | 274.40 | 264.97 | 263.93 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 13:15:00 | 263.30 | 268.88 | 268.88 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 11:15:00 | 269.90 | 268.56 | 268.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 09:15:00 | 285.80 | 272.51 | 270.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 282.25 | 282.39 | 277.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 09:30:00 | 284.25 | 282.39 | 277.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 279.40 | 281.82 | 278.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 279.40 | 281.82 | 278.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 279.80 | 281.41 | 278.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 279.65 | 281.41 | 278.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 278.00 | 280.43 | 278.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 278.00 | 280.43 | 278.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 280.20 | 280.39 | 278.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 282.40 | 280.39 | 278.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:45:00 | 281.40 | 280.53 | 278.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 15:00:00 | 281.00 | 280.30 | 279.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 09:15:00 | 276.85 | 279.86 | 279.34 | SL hit (close<static) qty=1.00 sl=277.50 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 11:15:00 | 277.60 | 278.96 | 278.99 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 279.45 | 279.08 | 279.03 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 10:15:00 | 278.40 | 278.94 | 278.97 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 280.35 | 279.22 | 279.10 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 277.00 | 278.87 | 279.00 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 286.00 | 280.29 | 279.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 13:15:00 | 291.40 | 285.07 | 282.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 09:15:00 | 293.50 | 299.45 | 294.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 293.50 | 299.45 | 294.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 293.50 | 299.45 | 294.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:45:00 | 293.85 | 299.45 | 294.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 292.00 | 297.96 | 294.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:30:00 | 292.25 | 297.96 | 294.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 10:15:00 | 289.90 | 292.57 | 292.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 12:15:00 | 289.20 | 291.69 | 292.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 12:15:00 | 288.70 | 288.30 | 289.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 13:00:00 | 288.70 | 288.30 | 289.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 289.95 | 288.63 | 289.53 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 292.55 | 290.31 | 290.15 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 287.00 | 289.72 | 289.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 285.60 | 288.89 | 289.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 283.40 | 282.79 | 285.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 12:45:00 | 283.35 | 282.79 | 285.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 284.95 | 281.07 | 282.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 284.95 | 281.07 | 282.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 282.55 | 281.36 | 282.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 11:45:00 | 281.85 | 281.47 | 282.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:45:00 | 282.05 | 282.00 | 282.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 15:00:00 | 281.05 | 281.81 | 282.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 290.65 | 283.53 | 282.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 290.65 | 283.53 | 282.93 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 281.60 | 283.68 | 283.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 14:15:00 | 280.50 | 282.22 | 283.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 287.15 | 282.93 | 283.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 287.15 | 282.93 | 283.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 287.15 | 282.93 | 283.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:00:00 | 287.15 | 282.93 | 283.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 288.80 | 284.10 | 283.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 09:15:00 | 307.55 | 290.35 | 287.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-05 09:15:00 | 314.55 | 314.72 | 303.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-05 09:30:00 | 316.05 | 314.72 | 303.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 310.80 | 312.53 | 308.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 12:45:00 | 310.85 | 312.53 | 308.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 313.75 | 315.29 | 313.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:15:00 | 317.25 | 315.29 | 313.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 11:45:00 | 316.60 | 314.63 | 313.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 09:45:00 | 317.75 | 314.53 | 313.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 15:15:00 | 312.00 | 313.35 | 313.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 312.00 | 313.35 | 313.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 309.00 | 311.53 | 312.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 10:15:00 | 307.45 | 306.36 | 308.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 10:15:00 | 307.45 | 306.36 | 308.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 307.45 | 306.36 | 308.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:00:00 | 307.45 | 306.36 | 308.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 309.30 | 306.94 | 308.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 308.70 | 306.94 | 308.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 308.20 | 307.20 | 308.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 13:15:00 | 307.95 | 307.20 | 308.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 13:15:00 | 314.05 | 308.57 | 309.19 | SL hit (close>static) qty=1.00 sl=310.85 alert=retest2 |

### Cycle 22 — BUY (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 14:15:00 | 316.85 | 310.22 | 309.89 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 11:15:00 | 309.10 | 310.60 | 310.62 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 314.00 | 310.75 | 310.59 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 307.75 | 310.65 | 310.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 13:15:00 | 305.25 | 308.54 | 309.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 307.55 | 307.37 | 308.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-24 09:45:00 | 307.60 | 307.37 | 308.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 307.15 | 306.65 | 307.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 14:00:00 | 307.15 | 306.65 | 307.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 308.30 | 306.54 | 307.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:30:00 | 314.70 | 306.54 | 307.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 307.40 | 306.71 | 307.52 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 13:15:00 | 309.70 | 308.27 | 308.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 311.85 | 308.98 | 308.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 12:15:00 | 309.90 | 310.34 | 309.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 13:00:00 | 309.90 | 310.34 | 309.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 308.70 | 309.92 | 309.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 14:45:00 | 307.10 | 309.92 | 309.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 309.50 | 309.84 | 309.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:15:00 | 307.70 | 309.84 | 309.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 310.00 | 309.87 | 309.48 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 307.80 | 309.14 | 309.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 303.65 | 307.44 | 308.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 310.45 | 303.97 | 305.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 310.45 | 303.97 | 305.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 310.45 | 303.97 | 305.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 310.45 | 303.97 | 305.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 311.40 | 305.46 | 306.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 311.40 | 305.46 | 306.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 311.95 | 306.76 | 306.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 13:15:00 | 316.00 | 311.90 | 309.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 314.20 | 317.17 | 314.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 13:15:00 | 314.20 | 317.17 | 314.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 314.20 | 317.17 | 314.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:45:00 | 315.00 | 317.17 | 314.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 318.90 | 317.51 | 314.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:30:00 | 313.80 | 317.51 | 314.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 317.70 | 317.32 | 315.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 11:00:00 | 320.50 | 318.91 | 317.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 11:30:00 | 320.80 | 319.45 | 317.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 322.40 | 319.17 | 318.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 10:00:00 | 320.50 | 319.44 | 318.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 315.40 | 318.65 | 318.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:30:00 | 315.45 | 318.65 | 318.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 317.95 | 318.51 | 318.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:00:00 | 317.95 | 318.51 | 318.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-07 13:15:00 | 315.10 | 317.83 | 317.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 13:15:00 | 315.10 | 317.83 | 317.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 12:15:00 | 314.85 | 316.77 | 317.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 315.35 | 314.75 | 316.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-09 10:00:00 | 315.35 | 314.75 | 316.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 306.95 | 313.19 | 315.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 09:15:00 | 306.00 | 309.36 | 310.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 09:45:00 | 305.85 | 308.91 | 310.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 10:45:00 | 305.90 | 308.42 | 310.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 11:15:00 | 305.60 | 308.42 | 310.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 306.00 | 306.15 | 307.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:45:00 | 306.30 | 306.15 | 307.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 309.60 | 306.84 | 307.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 309.60 | 306.84 | 307.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 310.15 | 307.50 | 308.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:30:00 | 310.10 | 307.50 | 308.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 308.40 | 308.44 | 308.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:30:00 | 308.55 | 308.44 | 308.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-17 11:15:00 | 309.00 | 308.55 | 308.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 11:15:00 | 309.00 | 308.55 | 308.52 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 13:15:00 | 305.00 | 307.84 | 308.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 15:15:00 | 303.35 | 306.49 | 307.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 11:15:00 | 309.80 | 306.70 | 307.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 11:15:00 | 309.80 | 306.70 | 307.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 309.80 | 306.70 | 307.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:00:00 | 309.80 | 306.70 | 307.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 12:15:00 | 316.25 | 308.61 | 308.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 15:15:00 | 320.40 | 313.56 | 310.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 09:15:00 | 311.45 | 313.14 | 310.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-21 09:45:00 | 312.00 | 313.14 | 310.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 312.00 | 312.94 | 311.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 313.95 | 312.94 | 311.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 315.90 | 313.53 | 312.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 12:00:00 | 316.80 | 314.08 | 312.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 10:30:00 | 318.80 | 316.11 | 314.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-11 14:15:00 | 336.60 | 339.36 | 339.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 14:15:00 | 336.60 | 339.36 | 339.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 333.15 | 336.85 | 338.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 337.05 | 336.19 | 337.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 337.05 | 336.19 | 337.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 337.05 | 336.19 | 337.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 337.05 | 336.19 | 337.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 336.90 | 335.06 | 336.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 334.90 | 335.06 | 336.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 335.60 | 335.17 | 336.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 338.50 | 335.17 | 336.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 336.40 | 335.42 | 336.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:45:00 | 336.80 | 335.42 | 336.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 341.35 | 336.60 | 336.82 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 342.60 | 337.80 | 337.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 343.20 | 340.13 | 338.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 340.75 | 341.95 | 340.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 340.75 | 341.95 | 340.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 340.75 | 341.95 | 340.62 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 336.95 | 339.95 | 340.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 330.30 | 337.87 | 339.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 330.10 | 327.58 | 330.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 10:15:00 | 330.10 | 327.58 | 330.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 330.10 | 327.58 | 330.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:45:00 | 330.85 | 327.58 | 330.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 330.60 | 328.19 | 330.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 14:30:00 | 327.65 | 329.22 | 330.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 10:00:00 | 329.00 | 329.22 | 330.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 11:30:00 | 328.70 | 329.32 | 329.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 12:45:00 | 328.90 | 329.26 | 329.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 328.75 | 329.13 | 329.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:30:00 | 330.35 | 329.13 | 329.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 327.15 | 328.72 | 329.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 10:30:00 | 325.00 | 328.02 | 329.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 15:15:00 | 323.90 | 326.50 | 327.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 12:15:00 | 329.20 | 327.38 | 327.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 12:15:00 | 329.20 | 327.38 | 327.25 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 326.50 | 327.11 | 327.15 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 329.50 | 327.49 | 327.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 330.45 | 328.09 | 327.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 327.70 | 328.62 | 328.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 14:15:00 | 327.70 | 328.62 | 328.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 327.70 | 328.62 | 328.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 327.70 | 328.62 | 328.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 327.40 | 328.37 | 327.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 339.90 | 328.37 | 327.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 13:15:00 | 330.40 | 334.15 | 333.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 330.45 | 333.18 | 333.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 14:15:00 | 330.05 | 332.55 | 332.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 330.05 | 332.55 | 332.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 09:15:00 | 327.90 | 331.36 | 332.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 15:15:00 | 330.00 | 329.80 | 330.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:15:00 | 325.35 | 329.80 | 330.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 324.90 | 323.21 | 324.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:45:00 | 325.80 | 323.21 | 324.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 325.30 | 323.63 | 325.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 325.35 | 323.63 | 325.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 325.60 | 324.02 | 325.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 13:45:00 | 326.30 | 324.02 | 325.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 324.90 | 324.20 | 325.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:45:00 | 325.00 | 324.20 | 325.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 326.70 | 324.70 | 325.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 12:00:00 | 324.85 | 324.84 | 325.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 09:15:00 | 339.25 | 326.70 | 325.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 09:15:00 | 339.25 | 326.70 | 325.68 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 11:15:00 | 330.70 | 335.06 | 335.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 13:15:00 | 329.90 | 333.25 | 334.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 334.10 | 332.45 | 333.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 334.10 | 332.45 | 333.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 334.10 | 332.45 | 333.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:00:00 | 334.10 | 332.45 | 333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 333.00 | 332.56 | 333.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 333.00 | 332.56 | 333.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 330.55 | 332.16 | 333.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 13:30:00 | 330.05 | 331.61 | 332.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 14:30:00 | 330.30 | 331.81 | 332.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:30:00 | 330.20 | 331.41 | 332.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 11:00:00 | 330.10 | 331.14 | 332.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 332.85 | 331.14 | 331.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 332.85 | 331.14 | 331.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 331.00 | 331.11 | 331.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:30:00 | 330.00 | 330.99 | 331.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:15:00 | 330.05 | 330.99 | 331.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:00:00 | 326.55 | 328.74 | 330.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:45:00 | 330.15 | 329.13 | 330.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 331.75 | 329.66 | 330.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 12:00:00 | 331.75 | 329.66 | 330.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 330.45 | 329.81 | 330.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:15:00 | 330.00 | 329.81 | 330.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:45:00 | 329.60 | 329.86 | 330.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:15:00 | 330.00 | 330.01 | 330.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-25 09:15:00 | 332.20 | 330.45 | 330.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 09:15:00 | 332.20 | 330.45 | 330.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-25 14:15:00 | 359.75 | 337.20 | 333.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-26 12:15:00 | 338.65 | 341.55 | 337.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-26 12:30:00 | 338.90 | 341.55 | 337.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 333.55 | 339.95 | 337.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 13:45:00 | 330.85 | 339.95 | 337.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 334.90 | 338.94 | 337.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-27 09:30:00 | 341.50 | 337.67 | 336.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 12:15:00 | 331.05 | 335.90 | 336.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 12:15:00 | 331.05 | 335.90 | 336.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 11:15:00 | 330.35 | 332.61 | 334.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 09:15:00 | 330.70 | 330.46 | 331.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-01 09:45:00 | 330.70 | 330.46 | 331.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 331.20 | 330.32 | 330.90 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 340.85 | 332.46 | 331.60 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 10:15:00 | 335.25 | 336.42 | 336.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 11:15:00 | 333.90 | 335.91 | 336.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 11:15:00 | 334.35 | 333.71 | 334.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 11:15:00 | 334.35 | 333.71 | 334.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 334.35 | 333.71 | 334.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:00:00 | 334.35 | 333.71 | 334.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 338.80 | 334.73 | 335.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:00:00 | 338.80 | 334.73 | 335.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 338.05 | 335.39 | 335.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 14:15:00 | 339.80 | 336.27 | 335.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 09:15:00 | 334.00 | 338.08 | 337.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 334.00 | 338.08 | 337.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 334.00 | 338.08 | 337.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:30:00 | 333.50 | 338.08 | 337.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 334.05 | 337.27 | 337.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 11:00:00 | 334.05 | 337.27 | 337.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 11:15:00 | 333.35 | 336.49 | 336.77 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 13:15:00 | 337.00 | 336.44 | 336.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 14:15:00 | 345.70 | 338.29 | 337.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 334.75 | 338.65 | 337.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 334.75 | 338.65 | 337.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 334.75 | 338.65 | 337.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 334.75 | 338.65 | 337.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 336.55 | 338.23 | 337.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 13:15:00 | 339.25 | 337.56 | 337.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 15:15:00 | 338.20 | 337.64 | 337.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 13:45:00 | 338.15 | 338.60 | 338.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 14:45:00 | 338.30 | 338.48 | 338.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 337.95 | 338.38 | 338.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 338.60 | 338.38 | 338.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 337.45 | 338.19 | 338.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-22 10:15:00 | 336.50 | 337.85 | 337.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 10:15:00 | 336.50 | 337.85 | 337.94 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 11:15:00 | 339.35 | 338.15 | 338.07 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 14:15:00 | 337.70 | 337.96 | 337.99 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 339.45 | 338.20 | 338.09 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 337.25 | 337.88 | 337.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 12:15:00 | 336.85 | 337.67 | 337.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 339.45 | 337.63 | 337.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 339.45 | 337.63 | 337.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 339.45 | 337.63 | 337.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 340.15 | 337.63 | 337.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 340.95 | 338.29 | 338.03 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 334.55 | 337.60 | 337.91 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 364.95 | 342.06 | 339.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 388.15 | 351.28 | 344.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 13:15:00 | 381.35 | 381.97 | 370.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 14:00:00 | 381.35 | 381.97 | 370.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 401.40 | 403.88 | 401.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:15:00 | 400.85 | 403.88 | 401.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 400.80 | 403.26 | 401.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 15:00:00 | 404.35 | 402.71 | 401.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 397.75 | 401.47 | 401.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 397.75 | 401.47 | 401.50 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 402.35 | 401.65 | 401.57 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 400.50 | 401.42 | 401.48 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 14:15:00 | 403.00 | 401.74 | 401.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 405.30 | 402.74 | 402.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 14:15:00 | 402.30 | 403.77 | 403.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 14:15:00 | 402.30 | 403.77 | 403.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 402.30 | 403.77 | 403.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:30:00 | 401.25 | 403.77 | 403.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 400.20 | 403.05 | 402.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:15:00 | 410.00 | 403.05 | 402.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 13:15:00 | 399.35 | 402.17 | 402.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2023-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 13:15:00 | 399.35 | 402.17 | 402.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 15:15:00 | 398.10 | 400.97 | 401.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 10:15:00 | 399.50 | 396.60 | 398.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 10:15:00 | 399.50 | 396.60 | 398.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 399.50 | 396.60 | 398.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 399.00 | 396.60 | 398.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 399.00 | 397.08 | 398.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:00:00 | 399.00 | 397.08 | 398.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 399.45 | 397.55 | 398.38 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 403.00 | 399.28 | 398.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 12:15:00 | 404.40 | 400.89 | 399.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 395.40 | 400.13 | 399.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 395.40 | 400.13 | 399.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 395.40 | 400.13 | 399.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 395.40 | 400.13 | 399.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 393.00 | 398.71 | 399.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 389.60 | 392.51 | 394.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 391.00 | 389.85 | 391.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 15:15:00 | 391.00 | 389.85 | 391.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 391.00 | 389.85 | 391.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 389.90 | 389.85 | 391.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 391.35 | 390.15 | 391.58 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 13:15:00 | 394.95 | 392.45 | 392.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 15:15:00 | 396.65 | 393.74 | 392.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 403.00 | 403.43 | 401.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 09:15:00 | 403.80 | 403.43 | 401.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 406.60 | 404.06 | 402.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:30:00 | 412.40 | 405.49 | 402.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 412.00 | 407.46 | 406.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 11:15:00 | 404.95 | 406.26 | 406.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 11:15:00 | 404.95 | 406.26 | 406.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 12:15:00 | 403.20 | 405.65 | 406.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 404.10 | 401.82 | 403.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 404.10 | 401.82 | 403.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 404.10 | 401.82 | 403.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:15:00 | 414.50 | 401.82 | 403.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 419.35 | 405.33 | 404.53 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 409.35 | 410.56 | 410.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 408.35 | 410.03 | 410.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 11:15:00 | 409.70 | 409.48 | 410.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 11:15:00 | 409.70 | 409.48 | 410.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 409.70 | 409.48 | 410.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:00:00 | 409.70 | 409.48 | 410.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 408.10 | 408.49 | 409.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 408.10 | 408.49 | 409.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 405.00 | 407.82 | 408.91 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 15:15:00 | 414.00 | 409.88 | 409.37 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 13:15:00 | 407.10 | 409.04 | 409.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 14:15:00 | 404.25 | 408.08 | 408.77 | Break + close below crossover candle low |

### Cycle 70 — BUY (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 09:15:00 | 430.05 | 407.94 | 407.16 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 427.80 | 432.67 | 432.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 424.00 | 430.93 | 431.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 15:15:00 | 429.00 | 427.48 | 429.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 15:15:00 | 429.00 | 427.48 | 429.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 429.00 | 427.48 | 429.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 429.75 | 427.48 | 429.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 431.50 | 428.28 | 429.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:45:00 | 431.20 | 428.28 | 429.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 430.35 | 428.70 | 429.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 11:15:00 | 425.45 | 428.70 | 429.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 09:15:00 | 432.35 | 429.92 | 429.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 432.35 | 429.92 | 429.91 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 425.00 | 429.31 | 429.70 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 431.45 | 428.86 | 428.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 434.80 | 430.41 | 429.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 436.70 | 438.16 | 435.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 12:00:00 | 436.70 | 438.16 | 435.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 434.30 | 437.39 | 434.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:00:00 | 434.30 | 437.39 | 434.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 433.60 | 436.63 | 434.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 432.95 | 436.63 | 434.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 432.20 | 435.75 | 434.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:45:00 | 433.00 | 435.75 | 434.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 433.00 | 435.20 | 434.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 434.65 | 435.20 | 434.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:15:00 | 435.20 | 435.07 | 434.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:45:00 | 435.45 | 435.20 | 434.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:30:00 | 434.60 | 435.04 | 434.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 431.20 | 434.27 | 434.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-02 12:15:00 | 431.20 | 434.27 | 434.25 | SL hit (close<static) qty=1.00 sl=431.50 alert=retest2 |

### Cycle 75 — SELL (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 13:15:00 | 429.70 | 433.36 | 433.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 427.10 | 430.95 | 432.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 436.65 | 428.92 | 430.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 436.65 | 428.92 | 430.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 436.65 | 428.92 | 430.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 439.70 | 428.92 | 430.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 441.40 | 431.41 | 431.23 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 436.00 | 437.63 | 437.70 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 445.00 | 438.40 | 437.93 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 14:15:00 | 440.40 | 442.14 | 442.15 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 443.80 | 442.47 | 442.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 462.40 | 446.46 | 444.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 469.45 | 470.14 | 461.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 13:00:00 | 469.45 | 470.14 | 461.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 463.90 | 468.50 | 463.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 464.85 | 468.50 | 463.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 466.65 | 468.13 | 463.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:45:00 | 469.15 | 467.27 | 465.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:30:00 | 468.90 | 467.64 | 465.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 13:15:00 | 469.45 | 472.02 | 472.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 469.45 | 472.02 | 472.16 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 473.75 | 472.34 | 472.28 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 09:15:00 | 466.90 | 471.25 | 471.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 12:15:00 | 464.65 | 468.13 | 470.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 10:15:00 | 466.25 | 460.88 | 463.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 10:15:00 | 466.25 | 460.88 | 463.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 466.25 | 460.88 | 463.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 11:00:00 | 466.25 | 460.88 | 463.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 478.90 | 464.49 | 465.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 11:30:00 | 478.95 | 464.49 | 465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 475.75 | 466.74 | 465.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 09:15:00 | 480.90 | 472.55 | 469.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 09:15:00 | 476.85 | 481.54 | 476.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 476.85 | 481.54 | 476.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 476.85 | 481.54 | 476.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:00:00 | 476.85 | 481.54 | 476.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 476.20 | 480.47 | 476.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:30:00 | 475.45 | 480.47 | 476.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 478.00 | 479.97 | 476.65 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 09:15:00 | 460.95 | 472.79 | 474.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 10:15:00 | 452.55 | 468.74 | 472.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 470.00 | 465.14 | 468.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 470.00 | 465.14 | 468.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 470.00 | 465.14 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:30:00 | 470.00 | 465.14 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 467.00 | 465.51 | 468.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 470.00 | 465.51 | 468.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 469.90 | 466.39 | 468.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 456.40 | 466.39 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 461.75 | 465.46 | 467.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 10:00:00 | 455.45 | 460.82 | 463.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 13:15:00 | 432.68 | 438.17 | 443.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 427.45 | 425.28 | 431.60 | SL hit (close>ema200) qty=0.50 sl=425.28 alert=retest2 |

### Cycle 86 — BUY (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 10:15:00 | 429.95 | 426.10 | 425.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 12:15:00 | 431.20 | 427.84 | 426.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 10:15:00 | 430.00 | 430.11 | 428.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 428.75 | 430.69 | 429.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 428.75 | 430.69 | 429.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 428.75 | 430.69 | 429.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 428.70 | 430.29 | 429.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 426.20 | 430.29 | 429.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 425.85 | 429.40 | 429.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:00:00 | 425.85 | 429.40 | 429.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 428.00 | 429.12 | 429.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 13:30:00 | 429.00 | 429.22 | 429.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 14:00:00 | 429.60 | 429.22 | 429.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 428.30 | 429.03 | 429.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 428.30 | 429.03 | 429.06 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 15:15:00 | 429.25 | 429.08 | 429.07 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 428.55 | 428.97 | 429.03 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 10:15:00 | 429.70 | 429.12 | 429.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 12:15:00 | 432.40 | 430.01 | 429.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 10:15:00 | 438.10 | 438.59 | 436.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 11:00:00 | 438.10 | 438.59 | 436.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 438.00 | 438.94 | 437.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 433.90 | 438.94 | 437.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 435.00 | 438.15 | 437.02 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 09:15:00 | 410.70 | 432.70 | 435.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 10:15:00 | 405.70 | 427.30 | 432.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 408.20 | 408.19 | 412.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 14:45:00 | 407.20 | 408.19 | 412.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 409.50 | 408.35 | 411.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 12:30:00 | 406.95 | 408.60 | 411.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 12:15:00 | 413.45 | 411.70 | 411.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 12:15:00 | 413.45 | 411.70 | 411.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 14:15:00 | 419.95 | 413.69 | 412.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 477.45 | 479.54 | 466.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 11:00:00 | 477.45 | 479.54 | 466.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 522.60 | 522.69 | 517.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 526.80 | 521.74 | 519.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 514.95 | 521.16 | 520.25 | SL hit (close<static) qty=1.00 sl=517.25 alert=retest2 |

### Cycle 93 — SELL (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 09:15:00 | 408.95 | 497.72 | 509.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 382.75 | 417.65 | 455.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 14:15:00 | 359.20 | 358.54 | 377.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 359.20 | 358.54 | 377.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 350.00 | 347.66 | 351.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:45:00 | 350.00 | 347.66 | 351.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 348.70 | 347.98 | 350.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:30:00 | 348.50 | 347.98 | 350.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 349.50 | 348.29 | 350.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:30:00 | 350.25 | 348.29 | 350.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 348.80 | 348.39 | 349.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:00:00 | 348.35 | 348.38 | 349.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 13:15:00 | 348.95 | 346.34 | 346.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 348.95 | 346.34 | 346.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 15:15:00 | 350.20 | 347.60 | 346.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 348.25 | 348.83 | 347.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 13:00:00 | 348.25 | 348.83 | 347.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 344.55 | 348.02 | 347.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 344.55 | 348.02 | 347.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 342.75 | 346.96 | 347.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 340.70 | 343.36 | 344.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 12:15:00 | 342.50 | 342.14 | 343.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 13:00:00 | 342.50 | 342.14 | 343.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 345.65 | 342.80 | 343.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 346.20 | 342.80 | 343.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 345.75 | 343.39 | 343.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:30:00 | 345.55 | 343.39 | 343.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 346.50 | 344.01 | 343.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 348.10 | 345.24 | 344.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 348.50 | 348.96 | 347.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 347.95 | 348.75 | 347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 348.15 | 348.75 | 347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 349.70 | 348.94 | 347.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 351.00 | 348.94 | 347.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 352.00 | 348.96 | 348.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 14:15:00 | 366.90 | 372.05 | 372.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 14:15:00 | 366.90 | 372.05 | 372.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 365.00 | 370.64 | 371.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 365.00 | 355.34 | 359.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 362.80 | 356.83 | 359.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 358.45 | 356.83 | 359.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 347.75 | 339.63 | 342.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 347.75 | 339.63 | 342.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 349.30 | 341.57 | 343.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 349.30 | 341.57 | 343.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 350.55 | 344.73 | 344.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 351.05 | 345.99 | 344.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 359.65 | 359.96 | 356.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 355.60 | 359.96 | 356.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 360.30 | 360.02 | 357.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 356.75 | 360.02 | 357.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 356.80 | 358.95 | 357.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 361.05 | 358.95 | 357.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 360.50 | 361.76 | 361.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:45:00 | 360.15 | 361.51 | 360.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 359.95 | 360.98 | 360.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 13:15:00 | 358.85 | 360.56 | 360.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 358.85 | 360.56 | 360.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 357.40 | 359.64 | 360.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 15:15:00 | 353.30 | 353.28 | 355.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 09:15:00 | 351.30 | 353.28 | 355.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 355.10 | 353.50 | 354.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 355.10 | 353.50 | 354.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 354.90 | 353.78 | 354.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 355.45 | 353.78 | 354.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 354.95 | 354.01 | 354.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:45:00 | 354.85 | 354.01 | 354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 352.65 | 353.74 | 354.63 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 360.95 | 355.66 | 355.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 15:15:00 | 380.20 | 364.54 | 360.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:45:00 | 366.75 | 374.40 | 368.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 364.00 | 372.32 | 368.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 354.85 | 372.32 | 368.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 355.05 | 363.56 | 364.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 352.75 | 354.61 | 357.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 349.50 | 348.11 | 351.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 11:45:00 | 350.00 | 348.11 | 351.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 350.10 | 347.71 | 349.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 350.10 | 347.71 | 349.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 351.35 | 348.44 | 349.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 352.15 | 348.44 | 349.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 350.60 | 349.31 | 349.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 349.20 | 349.48 | 349.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 15:15:00 | 341.35 | 340.84 | 340.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 341.35 | 340.84 | 340.81 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 340.05 | 340.68 | 340.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 337.25 | 339.09 | 339.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 12:15:00 | 335.10 | 335.07 | 336.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 13:00:00 | 335.10 | 335.07 | 336.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 335.65 | 335.18 | 336.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 335.65 | 335.18 | 336.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 333.00 | 334.66 | 335.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 333.45 | 334.66 | 335.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 325.90 | 324.15 | 326.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 325.90 | 324.15 | 326.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 326.55 | 324.63 | 326.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 327.20 | 324.63 | 326.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 327.15 | 325.13 | 326.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 327.15 | 325.13 | 326.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 328.00 | 325.71 | 326.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 328.35 | 325.71 | 326.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 328.05 | 326.76 | 327.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 324.40 | 327.37 | 327.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 328.65 | 327.62 | 327.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 328.65 | 327.62 | 327.60 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 325.50 | 327.41 | 327.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 15:15:00 | 324.05 | 326.74 | 327.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 323.80 | 322.67 | 324.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 326.00 | 323.40 | 324.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 326.45 | 323.40 | 324.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 328.10 | 324.34 | 324.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 328.10 | 324.34 | 324.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 330.40 | 325.83 | 325.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 332.65 | 329.79 | 327.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 336.20 | 338.54 | 334.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 336.20 | 338.54 | 334.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 369.20 | 370.37 | 364.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 10:15:00 | 371.80 | 370.37 | 364.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 377.95 | 367.92 | 365.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-09 13:15:00 | 408.98 | 401.53 | 395.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 383.30 | 395.30 | 396.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 382.80 | 392.80 | 394.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 13:15:00 | 390.95 | 390.79 | 393.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:45:00 | 390.65 | 390.79 | 393.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 387.40 | 389.90 | 392.17 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 393.00 | 390.90 | 390.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 396.00 | 393.09 | 391.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 394.00 | 392.97 | 392.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 390.00 | 392.38 | 391.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 390.00 | 392.38 | 391.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 389.55 | 391.81 | 391.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 389.20 | 391.81 | 391.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 392.65 | 391.85 | 391.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 393.45 | 392.08 | 391.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 394.90 | 392.08 | 391.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 402.25 | 406.07 | 406.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 402.25 | 406.07 | 406.46 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 407.70 | 406.34 | 406.34 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 406.15 | 406.31 | 406.32 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 406.70 | 406.38 | 406.35 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 404.55 | 406.21 | 406.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 404.00 | 405.77 | 406.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 406.50 | 403.95 | 404.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 406.95 | 404.55 | 405.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 406.95 | 404.55 | 405.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 407.15 | 405.07 | 405.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 12:45:00 | 403.35 | 404.68 | 405.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 405.90 | 403.55 | 404.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 11:15:00 | 407.15 | 404.86 | 404.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 407.15 | 404.86 | 404.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 12:15:00 | 412.45 | 408.03 | 406.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 11:15:00 | 409.00 | 410.11 | 408.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-09 12:00:00 | 409.00 | 410.11 | 408.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 406.00 | 409.28 | 408.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 406.00 | 409.28 | 408.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 407.40 | 408.91 | 408.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 405.65 | 408.91 | 408.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 409.40 | 408.76 | 408.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 407.55 | 408.53 | 408.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 413.00 | 409.42 | 408.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:30:00 | 413.30 | 410.63 | 409.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 13:15:00 | 413.20 | 410.63 | 409.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 407.70 | 409.31 | 409.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 407.70 | 409.31 | 409.31 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 409.40 | 409.33 | 409.32 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 407.75 | 409.01 | 409.18 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 414.50 | 410.11 | 409.65 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 409.30 | 414.54 | 414.73 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 11:15:00 | 416.55 | 414.94 | 414.89 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 413.75 | 414.80 | 414.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 411.65 | 414.17 | 414.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 416.95 | 413.53 | 414.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 420.80 | 414.98 | 414.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 423.10 | 417.64 | 415.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 414.95 | 417.10 | 415.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 414.20 | 416.52 | 415.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 412.50 | 416.52 | 415.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 413.40 | 415.90 | 415.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 413.40 | 415.90 | 415.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 416.40 | 416.00 | 415.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 420.60 | 416.47 | 415.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:30:00 | 419.00 | 421.19 | 420.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 418.70 | 420.56 | 420.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 418.70 | 420.56 | 420.62 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 420.80 | 420.64 | 420.62 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 15:15:00 | 419.55 | 420.42 | 420.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 415.95 | 419.53 | 420.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 416.15 | 412.39 | 414.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 412.50 | 412.41 | 414.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 409.60 | 411.93 | 413.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:15:00 | 409.85 | 411.93 | 413.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 417.00 | 414.56 | 414.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 417.00 | 414.56 | 414.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 09:15:00 | 420.30 | 417.09 | 415.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 412.55 | 419.71 | 418.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 414.70 | 418.71 | 417.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 414.70 | 418.71 | 417.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 414.95 | 417.96 | 417.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 414.05 | 417.96 | 417.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 411.60 | 416.69 | 416.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 407.40 | 413.06 | 414.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 412.70 | 408.75 | 411.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 414.50 | 409.90 | 411.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 414.50 | 409.90 | 411.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 415.55 | 411.03 | 412.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 415.55 | 411.03 | 412.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 418.25 | 413.45 | 413.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 420.00 | 415.47 | 414.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 421.00 | 421.78 | 419.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:30:00 | 421.15 | 421.78 | 419.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 418.05 | 420.88 | 419.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 418.05 | 420.88 | 419.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 417.20 | 420.14 | 419.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 416.75 | 420.14 | 419.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 418.65 | 419.64 | 419.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 418.40 | 419.64 | 419.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 417.50 | 419.21 | 419.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 418.20 | 419.21 | 419.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 416.45 | 418.66 | 418.79 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 419.75 | 418.96 | 418.86 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 416.75 | 419.03 | 419.06 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 424.50 | 419.40 | 419.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 429.00 | 423.99 | 421.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 424.95 | 426.22 | 424.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 423.95 | 425.77 | 424.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 423.50 | 425.77 | 424.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 424.90 | 425.59 | 424.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:15:00 | 423.15 | 425.59 | 424.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 422.35 | 424.94 | 424.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 420.40 | 424.94 | 424.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 419.75 | 423.91 | 423.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 419.75 | 423.91 | 423.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 419.80 | 423.08 | 423.52 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 428.75 | 423.69 | 423.07 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 417.70 | 422.79 | 422.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 415.70 | 421.37 | 422.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:30:00 | 451.20 | 414.29 | 414.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 10:15:00 | 434.95 | 418.43 | 416.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 09:15:00 | 453.05 | 437.90 | 428.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 09:15:00 | 439.95 | 443.40 | 436.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 439.95 | 443.40 | 436.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 438.00 | 440.76 | 437.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 438.00 | 440.76 | 437.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 439.15 | 440.44 | 437.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 438.00 | 440.44 | 437.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 435.55 | 439.46 | 437.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:45:00 | 435.30 | 439.46 | 437.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 431.95 | 437.96 | 437.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 432.00 | 437.96 | 437.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 432.00 | 435.79 | 436.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 14:15:00 | 429.50 | 433.56 | 435.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 13:15:00 | 437.00 | 435.30 | 435.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 443.45 | 437.59 | 436.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 438.50 | 439.57 | 439.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 15:15:00 | 435.00 | 438.29 | 439.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 442.80 | 439.69 | 439.44 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 435.40 | 438.97 | 439.38 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 439.75 | 438.44 | 438.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 443.00 | 439.45 | 438.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 437.85 | 440.96 | 440.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 437.50 | 440.27 | 439.80 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 432.55 | 438.72 | 439.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 430.00 | 431.89 | 433.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 439.10 | 433.33 | 434.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 437.80 | 434.22 | 434.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 439.20 | 434.22 | 434.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 436.80 | 435.06 | 434.90 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 433.40 | 434.75 | 434.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 428.10 | 433.42 | 434.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:00:00 | 433.55 | 432.68 | 433.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 434.80 | 433.11 | 433.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 434.80 | 433.11 | 433.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 429.10 | 432.31 | 433.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 15:00:00 | 428.90 | 431.62 | 432.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 428.85 | 430.57 | 432.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 439.80 | 433.84 | 433.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 439.80 | 433.84 | 433.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 454.10 | 442.23 | 439.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 501.75 | 502.42 | 490.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 501.75 | 502.42 | 490.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 489.10 | 496.68 | 491.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 491.00 | 496.68 | 491.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 483.35 | 494.02 | 490.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 483.35 | 494.02 | 490.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 482.75 | 488.92 | 489.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 09:15:00 | 481.50 | 486.44 | 488.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 14:15:00 | 485.35 | 484.40 | 486.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 485.35 | 484.40 | 486.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 494.90 | 486.44 | 486.84 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 489.90 | 487.66 | 487.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 15:15:00 | 492.25 | 489.70 | 488.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 11:15:00 | 490.15 | 490.28 | 489.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:00:00 | 490.15 | 490.28 | 489.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 489.30 | 490.09 | 489.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:00:00 | 489.30 | 490.09 | 489.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 491.45 | 490.36 | 489.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:30:00 | 490.15 | 490.36 | 489.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 490.50 | 490.61 | 489.88 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 485.70 | 489.41 | 489.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 484.15 | 486.80 | 487.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 490.60 | 486.84 | 487.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 489.00 | 487.27 | 487.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 487.50 | 487.27 | 487.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 486.85 | 487.19 | 487.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 484.75 | 486.57 | 487.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:30:00 | 485.05 | 486.10 | 487.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:00:00 | 484.20 | 486.10 | 487.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 484.90 | 480.34 | 481.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 485.95 | 481.46 | 481.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 484.15 | 482.00 | 481.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 484.15 | 482.00 | 481.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 15:15:00 | 489.70 | 484.76 | 483.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 12:15:00 | 491.25 | 492.30 | 489.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 491.25 | 492.30 | 489.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 490.25 | 491.89 | 489.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 490.35 | 491.89 | 489.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 492.55 | 491.95 | 490.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 500.55 | 491.95 | 490.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 485.80 | 494.12 | 492.76 | SL hit (close<static) qty=1.00 sl=489.90 alert=retest2 |

### Cycle 151 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 520.60 | 523.21 | 523.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 516.75 | 521.10 | 522.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 497.25 | 496.61 | 502.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 497.25 | 496.61 | 502.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 500.45 | 497.96 | 501.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 500.45 | 497.96 | 501.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 497.35 | 497.91 | 500.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 491.95 | 496.49 | 499.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 492.45 | 495.68 | 499.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 492.30 | 496.65 | 497.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 492.80 | 496.05 | 497.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 496.30 | 495.73 | 496.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 496.30 | 495.73 | 496.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 499.20 | 496.43 | 496.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 497.70 | 496.43 | 496.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 499.25 | 496.99 | 497.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 495.75 | 496.99 | 497.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 494.80 | 496.55 | 496.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 497.90 | 496.83 | 496.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 10:15:00 | 497.90 | 496.83 | 496.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 14:15:00 | 499.30 | 498.04 | 497.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 491.90 | 497.83 | 497.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 496.35 | 497.54 | 497.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 494.25 | 497.54 | 497.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 498.50 | 497.73 | 497.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 13:00:00 | 500.00 | 498.21 | 497.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:00:00 | 500.00 | 498.69 | 498.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:45:00 | 499.90 | 499.02 | 498.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 499.85 | 499.06 | 498.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 500.00 | 499.32 | 498.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 499.45 | 499.32 | 498.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 498.70 | 499.20 | 498.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 497.70 | 499.20 | 498.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 498.05 | 498.97 | 498.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 498.05 | 498.97 | 498.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 496.25 | 498.42 | 498.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 496.25 | 498.42 | 498.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 495.20 | 497.78 | 498.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 488.00 | 485.99 | 490.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 488.00 | 485.99 | 490.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 488.65 | 486.52 | 490.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 481.00 | 486.52 | 490.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 11:00:00 | 485.25 | 485.58 | 489.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 486.00 | 482.31 | 481.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 486.00 | 482.31 | 481.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 491.00 | 484.04 | 482.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:45:00 | 466.05 | 485.37 | 484.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 474.30 | 483.15 | 483.90 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 486.00 | 478.17 | 477.36 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 475.75 | 478.27 | 478.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 473.35 | 477.28 | 478.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 433.75 | 432.43 | 441.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 437.00 | 432.43 | 441.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 442.20 | 435.51 | 441.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 442.20 | 435.51 | 441.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 437.80 | 435.97 | 441.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 442.40 | 435.97 | 441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 411.50 | 406.60 | 410.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 411.50 | 406.60 | 410.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 416.00 | 408.48 | 411.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:00:00 | 416.00 | 408.48 | 411.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 416.20 | 410.02 | 411.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 416.80 | 410.02 | 411.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 417.00 | 413.38 | 413.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 12:15:00 | 420.00 | 416.77 | 415.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 417.15 | 417.46 | 415.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:15:00 | 426.30 | 417.46 | 415.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 417.80 | 417.53 | 416.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 417.80 | 417.53 | 416.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 418.85 | 417.80 | 416.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 415.00 | 417.30 | 416.55 | SL hit (close<ema400) qty=1.00 sl=416.55 alert=retest1 |

### Cycle 159 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 403.70 | 413.89 | 415.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 401.05 | 404.83 | 407.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 403.00 | 402.12 | 405.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 12:00:00 | 403.00 | 402.12 | 405.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 404.80 | 402.36 | 404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 404.80 | 402.36 | 404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 403.95 | 402.68 | 404.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 395.75 | 402.68 | 404.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:15:00 | 397.60 | 397.28 | 398.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 409.95 | 399.68 | 399.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 409.95 | 399.68 | 399.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 419.05 | 405.05 | 401.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 417.85 | 419.47 | 415.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 417.85 | 419.47 | 415.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 425.65 | 420.31 | 416.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 435.00 | 420.31 | 416.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 427.90 | 421.70 | 417.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:00:00 | 428.00 | 424.12 | 421.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 429.15 | 430.95 | 430.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 428.45 | 430.45 | 430.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 428.45 | 430.45 | 430.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 427.65 | 429.89 | 430.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 431.00 | 429.27 | 429.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 426.75 | 428.77 | 429.46 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 430.75 | 429.45 | 429.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 433.50 | 430.47 | 429.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 432.05 | 432.54 | 431.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 430.80 | 432.54 | 431.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 433.00 | 432.64 | 431.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 439.30 | 434.53 | 433.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 437.20 | 438.94 | 437.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 434.95 | 436.97 | 437.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 434.95 | 436.97 | 437.09 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 437.65 | 437.22 | 437.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 447.45 | 439.26 | 438.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 478.05 | 476.24 | 470.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 477.35 | 476.24 | 470.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 477.30 | 476.31 | 471.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 477.30 | 481.10 | 478.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 479.75 | 480.83 | 478.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:45:00 | 483.40 | 481.22 | 478.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 13:15:00 | 483.90 | 481.22 | 478.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:15:00 | 483.85 | 481.50 | 479.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 483.00 | 481.80 | 479.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 464.20 | 478.48 | 478.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 464.20 | 478.48 | 478.29 | SL hit (close<static) qty=1.00 sl=465.05 alert=retest2 |

### Cycle 165 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 474.85 | 477.75 | 477.97 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 486.15 | 478.82 | 478.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 496.00 | 484.15 | 481.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:30:00 | 494.00 | 498.08 | 493.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 12:15:00 | 493.05 | 497.07 | 493.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:45:00 | 492.15 | 497.07 | 493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 491.90 | 496.04 | 493.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:15:00 | 491.50 | 496.04 | 493.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 490.35 | 494.90 | 493.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:30:00 | 490.85 | 494.90 | 493.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 497.85 | 495.14 | 493.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 501.20 | 496.35 | 494.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 501.90 | 498.12 | 495.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:30:00 | 500.65 | 499.24 | 497.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 502.00 | 499.61 | 497.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 507.15 | 506.26 | 503.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 508.20 | 506.26 | 503.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:00:00 | 507.85 | 507.34 | 504.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 511.85 | 507.93 | 505.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 502.65 | 508.68 | 507.54 | SL hit (close<static) qty=1.00 sl=503.30 alert=retest2 |

### Cycle 167 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 495.65 | 506.07 | 506.46 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 510.50 | 507.25 | 506.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 512.50 | 508.30 | 507.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 513.15 | 513.15 | 511.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:00:00 | 513.15 | 513.15 | 511.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 512.80 | 513.08 | 511.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 509.90 | 513.08 | 511.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 512.00 | 512.86 | 511.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 506.75 | 512.86 | 511.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 503.10 | 510.91 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 503.10 | 510.91 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 508.25 | 510.38 | 510.41 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 510.95 | 509.51 | 509.41 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 507.25 | 509.10 | 509.25 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 517.00 | 510.68 | 509.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 521.20 | 514.70 | 512.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 524.15 | 520.24 | 518.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 15:15:00 | 576.57 | 562.61 | 553.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 577.45 | 580.33 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 567.00 | 577.66 | 579.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 569.00 | 561.77 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 567.70 | 562.96 | 567.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 562.70 | 566.52 | 567.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 560.65 | 564.81 | 566.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 580.20 | 559.84 | 555.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 582.35 | 583.11 | 576.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 582.35 | 583.11 | 576.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 577.50 | 581.14 | 577.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 577.50 | 581.14 | 577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 577.50 | 580.41 | 577.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 573.40 | 580.41 | 577.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 572.00 | 578.73 | 577.05 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 569.00 | 575.67 | 575.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 565.80 | 569.34 | 571.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 571.00 | 569.67 | 571.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 571.55 | 570.05 | 571.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 571.55 | 570.05 | 571.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 574.10 | 570.86 | 571.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 570.30 | 570.75 | 571.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 569.80 | 570.56 | 571.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 561.50 | 570.41 | 571.30 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 580.00 | 570.24 | 569.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 589.30 | 574.05 | 571.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 572.95 | 581.03 | 577.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 571.20 | 579.07 | 576.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 571.20 | 579.07 | 576.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 566.35 | 574.35 | 574.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 563.60 | 567.92 | 569.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 562.40 | 562.28 | 565.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:15:00 | 563.70 | 562.28 | 565.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 569.85 | 563.79 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 570.65 | 563.79 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 569.90 | 565.01 | 565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 571.20 | 565.01 | 565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 570.25 | 566.78 | 566.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 573.25 | 568.07 | 567.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:30:00 | 579.90 | 574.47 | 571.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 592.60 | 590.48 | 590.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 600.20 | 592.43 | 591.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 592.40 | 594.23 | 592.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 594.00 | 594.18 | 592.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 620.15 | 594.18 | 592.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 612.50 | 618.35 | 620.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 605.40 | 603.83 | 608.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 12:00:00 | 605.40 | 603.83 | 608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 609.20 | 605.39 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 608.50 | 605.39 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 608.55 | 606.03 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 610.00 | 606.03 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 605.50 | 605.92 | 608.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 601.90 | 605.92 | 608.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 604.25 | 605.59 | 607.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 597.80 | 602.10 | 604.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 598.50 | 601.63 | 603.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 598.85 | 600.69 | 602.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:00:00 | 597.30 | 599.88 | 601.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 593.75 | 596.16 | 598.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 592.10 | 596.16 | 598.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 593.65 | 595.37 | 597.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 587.25 | 595.00 | 596.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 600.60 | 597.97 | 596.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 594.95 | 598.94 | 596.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 595.65 | 598.28 | 596.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:15:00 | 595.30 | 598.28 | 596.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 598.40 | 598.30 | 597.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:30:00 | 595.95 | 598.30 | 597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 593.95 | 597.46 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 594.60 | 597.46 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 596.90 | 597.35 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 598.00 | 597.35 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 598.30 | 597.54 | 596.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 599.45 | 597.54 | 596.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 599.40 | 598.03 | 597.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 599.55 | 598.34 | 597.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 595.40 | 597.75 | 597.28 | SL hit (close<static) qty=1.00 sl=596.05 alert=retest2 |

### Cycle 185 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 593.45 | 596.89 | 596.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 590.40 | 595.59 | 596.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 583.70 | 580.39 | 584.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 581.60 | 580.63 | 584.36 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 591.80 | 585.97 | 585.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 594.15 | 587.60 | 586.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 596.40 | 601.82 | 597.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 597.05 | 600.87 | 597.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:15:00 | 596.40 | 600.87 | 597.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 593.00 | 599.30 | 597.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 593.00 | 599.30 | 597.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 596.15 | 598.67 | 597.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 598.15 | 597.11 | 596.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 599.60 | 594.72 | 594.11 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 592.70 | 594.72 | 594.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 590.75 | 593.65 | 594.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 594.50 | 593.53 | 594.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 596.00 | 594.02 | 594.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 590.00 | 594.02 | 594.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 590.00 | 593.22 | 593.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 588.00 | 591.75 | 593.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 587.00 | 585.29 | 587.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 600.55 | 592.09 | 589.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 608.20 | 608.57 | 603.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 608.20 | 608.57 | 603.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 606.40 | 607.59 | 604.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 612.40 | 607.59 | 604.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 616.30 | 615.46 | 615.42 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 610.25 | 614.42 | 614.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 606.05 | 612.75 | 614.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 605.70 | 605.79 | 608.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 608.40 | 606.31 | 608.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 604.95 | 606.53 | 608.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 601.15 | 604.79 | 606.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 605.15 | 605.16 | 606.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 574.70 | 597.43 | 600.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 597.55 | 597.43 | 600.49 | SL hit (close>static) qty=0.50 sl=597.43 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 611.60 | 603.28 | 602.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 618.75 | 609.70 | 606.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 617.30 | 618.23 | 612.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:30:00 | 617.00 | 618.23 | 612.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 635.30 | 637.68 | 630.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 630.90 | 637.68 | 630.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 637.95 | 638.09 | 632.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 645.00 | 638.90 | 634.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 642.55 | 639.14 | 635.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 642.55 | 639.26 | 637.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 642.25 | 641.70 | 640.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 640.35 | 641.43 | 640.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 640.00 | 639.83 | 639.82 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 632.00 | 638.26 | 639.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 627.60 | 635.14 | 637.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 632.35 | 631.91 | 635.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 632.35 | 631.91 | 635.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 613.90 | 613.83 | 617.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 613.00 | 613.83 | 617.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 617.20 | 614.05 | 616.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 621.45 | 614.05 | 616.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 629.00 | 617.04 | 617.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 627.70 | 617.04 | 617.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 631.40 | 619.91 | 618.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 637.80 | 626.88 | 622.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 646.95 | 647.08 | 638.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 647.15 | 647.08 | 638.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 639.15 | 644.49 | 638.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 639.15 | 644.49 | 638.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 636.30 | 642.86 | 638.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 635.95 | 642.86 | 638.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 637.35 | 641.75 | 638.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 639.60 | 639.99 | 638.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 641.80 | 640.29 | 638.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 635.00 | 638.87 | 638.25 | SL hit (close<static) qty=1.00 sl=636.10 alert=retest2 |

### Cycle 199 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 633.70 | 637.83 | 637.84 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 646.65 | 637.59 | 637.30 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 629.30 | 636.47 | 637.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 623.65 | 633.91 | 636.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 640.80 | 631.63 | 633.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 629.00 | 631.11 | 633.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 626.70 | 631.11 | 633.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 627.70 | 629.98 | 632.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 626.65 | 628.82 | 631.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 627.40 | 627.03 | 629.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 626.00 | 626.83 | 628.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 639.70 | 632.14 | 630.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 663.55 | 667.57 | 657.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 663.55 | 667.57 | 657.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 658.80 | 665.10 | 658.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 667.25 | 665.10 | 658.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 695.35 | 692.55 | 692.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 699.00 | 693.84 | 692.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 696.45 | 697.05 | 695.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:45:00 | 696.95 | 697.05 | 695.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 700.70 | 717.61 | 715.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 700.70 | 717.61 | 715.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 701.15 | 714.32 | 714.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 700.25 | 714.32 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 703.60 | 712.17 | 713.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 696.25 | 703.24 | 707.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 699.90 | 699.37 | 702.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 699.90 | 699.37 | 702.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 696.15 | 698.31 | 701.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 693.95 | 698.72 | 700.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 694.75 | 697.58 | 699.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 678.85 | 697.75 | 699.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 691.20 | 686.36 | 690.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 692.00 | 687.49 | 690.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 692.50 | 687.49 | 690.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 683.55 | 686.70 | 689.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 681.85 | 686.70 | 689.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 680.10 | 682.79 | 686.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 12:15:00 | 697.50 | 690.20 | 688.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 686.65 | 690.39 | 688.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 689.00 | 690.11 | 688.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 679.45 | 690.11 | 688.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 681.45 | 688.38 | 688.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:45:00 | 694.15 | 689.39 | 688.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 673.25 | 683.40 | 686.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 15:15:00 | 682.80 | 682.21 | 684.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:15:00 | 672.35 | 682.21 | 684.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 679.55 | 679.01 | 681.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 679.55 | 679.01 | 681.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | SL hit (close>ema400) qty=1.00 sl=681.54 alert=retest1 |

### Cycle 208 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 685.00 | 682.66 | 682.58 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 678.00 | 681.69 | 682.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 675.20 | 680.39 | 681.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 681.90 | 680.69 | 681.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 678.40 | 680.24 | 681.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 675.30 | 678.24 | 680.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 676.80 | 677.66 | 679.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 676.05 | 677.87 | 679.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 676.40 | 677.89 | 679.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 677.40 | 677.79 | 678.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 678.20 | 677.79 | 678.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 678.35 | 677.90 | 678.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 678.80 | 677.90 | 678.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 678.10 | 677.94 | 678.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 673.35 | 677.94 | 678.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 674.65 | 677.00 | 678.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 675.90 | 676.19 | 677.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 680.50 | 677.05 | 677.87 | SL hit (close>static) qty=1.00 sl=679.00 alert=retest2 |

### Cycle 210 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 697.25 | 681.93 | 679.94 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 672.70 | 680.74 | 681.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 669.00 | 676.98 | 679.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 671.80 | 671.58 | 676.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 671.80 | 671.58 | 676.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 669.10 | 659.83 | 664.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 669.10 | 659.83 | 664.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 669.50 | 661.77 | 665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 669.50 | 661.77 | 665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 656.80 | 664.17 | 665.40 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 665.00 | 662.76 | 662.69 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 661.85 | 662.58 | 662.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 660.60 | 662.18 | 662.43 | Break + close below crossover candle low |

### Cycle 214 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 664.40 | 662.62 | 662.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 668.85 | 663.87 | 663.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 660.35 | 662.66 | 662.90 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 666.10 | 662.98 | 662.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 669.80 | 664.68 | 663.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 663.05 | 664.96 | 664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 665.15 | 665.00 | 664.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 666.85 | 666.00 | 664.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 658.10 | 665.20 | 664.75 | SL hit (close<static) qty=1.00 sl=662.00 alert=retest2 |

### Cycle 217 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 660.10 | 664.18 | 664.33 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 667.50 | 664.03 | 664.02 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 660.65 | 663.35 | 663.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 659.95 | 662.67 | 663.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 657.45 | 656.01 | 658.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 657.45 | 656.01 | 658.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 650.75 | 654.96 | 657.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 648.80 | 653.03 | 656.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 616.36 | 623.00 | 633.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 622.00 | 621.27 | 628.47 | SL hit (close>ema200) qty=0.50 sl=621.27 alert=retest2 |

### Cycle 220 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 610.70 | 599.73 | 598.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 616.05 | 604.89 | 601.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 607.10 | 612.59 | 607.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 608.25 | 611.72 | 607.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 613.30 | 611.95 | 608.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 614.00 | 612.48 | 608.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 605.85 | 610.73 | 608.87 | SL hit (close<static) qty=1.00 sl=606.55 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 600.00 | 607.71 | 607.77 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 615.85 | 608.10 | 607.86 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 604.55 | 610.02 | 610.19 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 612.00 | 609.51 | 609.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 614.60 | 610.53 | 609.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 618.00 | 620.44 | 617.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 618.00 | 620.44 | 617.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 618.05 | 619.96 | 617.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 616.40 | 619.96 | 617.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 620.60 | 620.09 | 617.85 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 615.35 | 617.36 | 617.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 613.30 | 615.89 | 616.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 609.65 | 607.60 | 607.60 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 606.90 | 607.61 | 607.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 600.60 | 605.83 | 606.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 560.30 | 555.84 | 566.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 559.25 | 555.84 | 566.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 568.25 | 558.99 | 566.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 568.25 | 558.99 | 566.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 567.20 | 560.63 | 566.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 571.75 | 560.63 | 566.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 572.60 | 564.52 | 567.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 572.60 | 564.52 | 567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 570.35 | 565.68 | 567.60 | EMA400 retest candle locked (from downside) |

### Cycle 228 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 574.90 | 569.69 | 569.07 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 565.05 | 568.33 | 568.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 561.30 | 566.93 | 567.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 557.75 | 556.48 | 559.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 557.75 | 556.48 | 559.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 548.30 | 542.91 | 547.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 548.30 | 542.91 | 547.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 545.55 | 543.44 | 546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:15:00 | 550.25 | 543.44 | 546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 550.00 | 544.75 | 547.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 550.95 | 544.75 | 547.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 551.95 | 546.19 | 547.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 551.95 | 546.19 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 558.50 | 550.08 | 549.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 566.55 | 553.38 | 550.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 563.50 | 557.51 | 556.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 15:15:00 | 544.55 | 552.51 | 554.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 538.30 | 536.81 | 541.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 538.30 | 536.81 | 541.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 549.80 | 539.36 | 541.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 552.95 | 539.36 | 541.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 558.00 | 543.09 | 543.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 558.00 | 543.09 | 543.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 560.00 | 546.47 | 544.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 562.00 | 551.75 | 547.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 592.45 | 599.52 | 593.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 597.10 | 599.04 | 593.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 607.85 | 601.64 | 597.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 608.00 | 604.28 | 598.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 636.40 | 643.11 | 644.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:45:00 | 645.45 | 643.18 | 644.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 649.80 | 644.50 | 644.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:45:00 | 648.35 | 644.50 | 644.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — BUY (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 13:15:00 | 646.00 | 644.80 | 644.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 656.05 | 648.49 | 646.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 643.45 | 665.56 | 661.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 643.70 | 661.19 | 659.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 637.35 | 661.19 | 659.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 660.95 | 659.83 | 659.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 665.05 | 660.07 | 659.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 667.60 | 662.57 | 660.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 651.00 | 667.48 | 672.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 670.10 | 655.14 | 661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 672.25 | 658.57 | 662.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 672.25 | 658.57 | 662.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 664.65 | 660.64 | 662.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 665.45 | 660.64 | 662.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 671.05 | 662.72 | 663.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 671.05 | 662.72 | 663.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 668.90 | 663.96 | 663.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 672.70 | 665.70 | 664.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 13:15:00 | 671.85 | 672.21 | 668.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 674.85 | 672.21 | 668.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 664.00 | 670.57 | 668.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 664.00 | 670.57 | 668.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 659.00 | 668.26 | 667.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 645.95 | 668.26 | 667.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 644.70 | 663.54 | 665.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 639.45 | 658.73 | 663.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 623.20 | 620.95 | 630.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 635.05 | 620.95 | 630.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.25 | 622.81 | 630.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 624.80 | 624.65 | 630.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 638.95 | 627.51 | 631.00 | SL hit (close>static) qty=1.00 sl=636.90 alert=retest2 |

### Cycle 238 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 641.60 | 634.11 | 633.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 650.80 | 637.45 | 634.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 651.00 | 656.84 | 650.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 650.80 | 655.63 | 650.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 645.65 | 655.63 | 650.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 652.55 | 655.01 | 650.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 653.00 | 655.01 | 650.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | EMA200 below EMA400 |

### Cycle 240 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 670.50 | 665.95 | 665.37 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 657.10 | 664.70 | 665.21 | EMA200 below EMA400 |

### Cycle 242 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 670.00 | 663.71 | 663.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 673.55 | 667.23 | 665.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 668.00 | 669.77 | 667.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 671.00 | 670.01 | 668.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 671.65 | 670.22 | 668.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 671.80 | 670.55 | 668.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:00:00 | 671.85 | 670.55 | 668.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 672.10 | 672.38 | 669.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 672.85 | 676.17 | 672.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 679.45 | 676.45 | 673.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 243 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 678.65 | 672.20 | 671.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 681.60 | 674.08 | 672.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 676.00 | 679.51 | 676.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 678.40 | 679.29 | 676.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 678.15 | 679.29 | 676.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 679.85 | 679.10 | 677.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:15:00 | 672.90 | 679.10 | 677.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 676.00 | 678.48 | 677.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:30:00 | 673.55 | 678.48 | 677.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 675.70 | 677.92 | 677.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 677.85 | 677.92 | 677.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 678.30 | 683.60 | 682.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 245 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 687.35 | 682.79 | 682.23 | EMA200 above EMA400 |

### Cycle 247 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 677.60 | 681.88 | 682.35 | EMA200 below EMA400 |

### Cycle 248 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 684.00 | 681.96 | 681.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 685.65 | 682.69 | 682.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 714.90 | 714.95 | 707.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:30:00 | 715.00 | 714.95 | 707.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 706.20 | 713.20 | 707.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 706.20 | 713.20 | 707.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 705.35 | 711.63 | 707.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 705.35 | 711.63 | 707.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 705.65 | 710.43 | 707.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 705.65 | 710.43 | 707.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 698.55 | 707.35 | 706.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 698.60 | 707.35 | 706.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 249 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 691.15 | 704.11 | 704.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 685.45 | 696.16 | 699.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 703.00 | 695.08 | 698.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 707.00 | 697.47 | 698.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 750.20 | 697.47 | 698.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 250 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 754.80 | 708.93 | 703.96 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 13:30:00 | 255.80 | 2023-05-19 13:15:00 | 255.35 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-05-17 09:15:00 | 256.40 | 2023-05-19 13:15:00 | 255.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-05-17 12:15:00 | 256.00 | 2023-05-19 13:15:00 | 255.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-05-19 12:45:00 | 256.25 | 2023-05-19 13:15:00 | 255.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-05-26 13:45:00 | 270.40 | 2023-05-29 11:15:00 | 263.55 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2023-05-29 10:15:00 | 269.35 | 2023-05-29 11:15:00 | 263.55 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2023-05-29 11:15:00 | 270.15 | 2023-05-29 11:15:00 | 263.55 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2023-06-09 09:15:00 | 282.40 | 2023-06-12 09:15:00 | 276.85 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2023-06-09 09:45:00 | 281.40 | 2023-06-12 09:15:00 | 276.85 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-06-09 15:00:00 | 281.00 | 2023-06-12 09:15:00 | 276.85 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-06-27 11:45:00 | 281.85 | 2023-06-28 09:15:00 | 290.65 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2023-06-27 13:45:00 | 282.05 | 2023-06-28 09:15:00 | 290.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2023-06-27 15:00:00 | 281.05 | 2023-06-28 09:15:00 | 290.65 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2023-07-10 09:15:00 | 317.25 | 2023-07-11 15:15:00 | 312.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-07-10 11:45:00 | 316.60 | 2023-07-11 15:15:00 | 312.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-07-11 09:45:00 | 317.75 | 2023-07-11 15:15:00 | 312.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2023-07-14 13:15:00 | 307.95 | 2023-07-14 13:15:00 | 314.05 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-08-04 11:00:00 | 320.50 | 2023-08-07 13:15:00 | 315.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-08-04 11:30:00 | 320.80 | 2023-08-07 13:15:00 | 315.10 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-08-07 09:15:00 | 322.40 | 2023-08-07 13:15:00 | 315.10 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-08-07 10:00:00 | 320.50 | 2023-08-07 13:15:00 | 315.10 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-08-14 09:15:00 | 306.00 | 2023-08-17 11:15:00 | 309.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-08-14 09:45:00 | 305.85 | 2023-08-17 11:15:00 | 309.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-08-14 10:45:00 | 305.90 | 2023-08-17 11:15:00 | 309.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-08-14 11:15:00 | 305.60 | 2023-08-17 11:15:00 | 309.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-08-22 12:00:00 | 316.80 | 2023-09-11 14:15:00 | 336.60 | STOP_HIT | 1.00 | 6.25% |
| BUY | retest2 | 2023-08-24 10:30:00 | 318.80 | 2023-09-11 14:15:00 | 336.60 | STOP_HIT | 1.00 | 5.58% |
| SELL | retest2 | 2023-09-22 14:30:00 | 327.65 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-09-25 10:00:00 | 329.00 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-09-25 11:30:00 | 328.70 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-09-25 12:45:00 | 328.90 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-09-26 10:30:00 | 325.00 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-09-27 15:15:00 | 323.90 | 2023-09-28 12:15:00 | 329.20 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-10-03 09:15:00 | 339.90 | 2023-10-04 14:15:00 | 330.05 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2023-10-04 13:15:00 | 330.40 | 2023-10-04 14:15:00 | 330.05 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-10-04 14:15:00 | 330.45 | 2023-10-04 14:15:00 | 330.05 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2023-10-11 12:00:00 | 324.85 | 2023-10-12 09:15:00 | 339.25 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2023-10-18 13:30:00 | 330.05 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-18 14:30:00 | 330.30 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-10-19 09:30:00 | 330.20 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-10-19 11:00:00 | 330.10 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-10-20 09:30:00 | 330.00 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-10-20 10:15:00 | 330.05 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-23 10:00:00 | 326.55 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2023-10-23 10:45:00 | 330.15 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-10-23 13:15:00 | 330.00 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-10-23 13:45:00 | 329.60 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-10-23 15:15:00 | 330.00 | 2023-10-25 09:15:00 | 332.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-10-27 09:30:00 | 341.50 | 2023-10-27 12:15:00 | 331.05 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2023-11-20 13:15:00 | 339.25 | 2023-11-22 10:15:00 | 336.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-11-20 15:15:00 | 338.20 | 2023-11-22 10:15:00 | 336.50 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-11-21 13:45:00 | 338.15 | 2023-11-22 10:15:00 | 336.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-11-21 14:45:00 | 338.30 | 2023-11-22 10:15:00 | 336.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-12-07 15:00:00 | 404.35 | 2023-12-08 11:15:00 | 397.75 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-12-12 09:15:00 | 410.00 | 2023-12-12 13:15:00 | 399.35 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2023-12-29 10:30:00 | 412.40 | 2024-01-03 11:15:00 | 404.95 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-01-02 09:30:00 | 412.00 | 2024-01-03 11:15:00 | 404.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-01-24 11:15:00 | 425.45 | 2024-01-25 09:15:00 | 432.35 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-02-02 09:15:00 | 434.65 | 2024-02-02 12:15:00 | 431.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-02-02 10:15:00 | 435.20 | 2024-02-02 12:15:00 | 431.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-02-02 10:45:00 | 435.45 | 2024-02-02 12:15:00 | 431.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-02-02 11:30:00 | 434.60 | 2024-02-02 12:15:00 | 431.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-02-20 09:45:00 | 469.15 | 2024-02-22 13:15:00 | 469.45 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-02-20 10:30:00 | 468.90 | 2024-02-22 13:15:00 | 469.45 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-03-05 10:00:00 | 455.45 | 2024-03-11 13:15:00 | 432.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 10:00:00 | 455.45 | 2024-03-13 09:15:00 | 427.45 | STOP_HIT | 0.50 | 6.15% |
| BUY | retest2 | 2024-03-19 13:30:00 | 429.00 | 2024-03-19 14:15:00 | 428.30 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-03-19 14:00:00 | 429.60 | 2024-03-19 14:15:00 | 428.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-04-02 12:30:00 | 406.95 | 2024-04-03 12:15:00 | 413.45 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-04-22 09:15:00 | 526.80 | 2024-04-22 14:15:00 | 514.95 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-05-06 13:00:00 | 348.35 | 2024-05-08 13:15:00 | 348.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-05-16 15:15:00 | 351.00 | 2024-05-29 14:15:00 | 366.90 | STOP_HIT | 1.00 | 4.53% |
| BUY | retest2 | 2024-05-21 09:15:00 | 352.00 | 2024-05-29 14:15:00 | 366.90 | STOP_HIT | 1.00 | 4.23% |
| BUY | retest2 | 2024-06-12 09:15:00 | 361.05 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-14 09:45:00 | 360.50 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-06-14 10:45:00 | 360.15 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-06-14 12:45:00 | 359.95 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-02 09:30:00 | 349.20 | 2024-07-10 15:15:00 | 341.35 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2024-07-23 12:15:00 | 324.40 | 2024-07-23 12:15:00 | 328.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-08-05 10:15:00 | 371.80 | 2024-08-09 13:15:00 | 408.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 377.95 | 2024-08-12 15:15:00 | 383.30 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-08-20 14:45:00 | 393.45 | 2024-08-29 14:15:00 | 402.25 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2024-08-20 15:15:00 | 394.90 | 2024-08-29 14:15:00 | 402.25 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2024-09-03 12:45:00 | 403.35 | 2024-09-04 11:15:00 | 407.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-04 10:15:00 | 405.90 | 2024-09-04 11:15:00 | 407.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-10 12:30:00 | 413.30 | 2024-09-11 12:15:00 | 407.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-09-10 13:15:00 | 413.20 | 2024-09-11 12:15:00 | 407.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-20 09:15:00 | 420.60 | 2024-09-25 10:15:00 | 418.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-09-24 12:30:00 | 419.00 | 2024-09-25 10:15:00 | 418.70 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-09-30 09:30:00 | 409.60 | 2024-10-01 10:15:00 | 417.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-09-30 10:15:00 | 409.85 | 2024-10-01 10:15:00 | 417.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-21 15:00:00 | 428.90 | 2024-11-25 10:15:00 | 439.80 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-11-22 10:15:00 | 428.85 | 2024-11-25 10:15:00 | 439.80 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-12-12 12:00:00 | 484.75 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-12-12 12:30:00 | 485.05 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-12-12 13:00:00 | 484.20 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-12-17 10:15:00 | 484.90 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-12-20 09:15:00 | 500.55 | 2024-12-20 14:15:00 | 485.80 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-12-23 09:15:00 | 494.10 | 2025-01-08 12:15:00 | 520.60 | STOP_HIT | 1.00 | 5.36% |
| SELL | retest2 | 2025-01-15 11:30:00 | 491.95 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-01-15 13:00:00 | 492.45 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-17 09:30:00 | 492.30 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-17 10:45:00 | 492.80 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-01-22 13:00:00 | 500.00 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-01-23 10:00:00 | 500.00 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-01-23 11:45:00 | 499.90 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-23 12:45:00 | 499.85 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-01-28 09:15:00 | 481.00 | 2025-01-30 15:15:00 | 486.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-01-28 11:00:00 | 485.25 | 2025-01-30 15:15:00 | 486.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-02-21 09:15:00 | 426.30 | 2025-02-21 14:15:00 | 415.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-03-03 09:15:00 | 395.75 | 2025-03-05 09:15:00 | 409.95 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-03-04 13:15:00 | 397.60 | 2025-03-05 09:15:00 | 409.95 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-03-10 10:15:00 | 435.00 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-03-10 10:45:00 | 427.90 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-03-11 12:00:00 | 428.00 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-03-17 13:00:00 | 429.15 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-03-25 09:30:00 | 439.30 | 2025-03-27 11:15:00 | 434.95 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-03-27 09:15:00 | 437.20 | 2025-03-27 11:15:00 | 434.95 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-04-02 12:30:00 | 478.05 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-04-02 13:00:00 | 477.35 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-04-02 14:15:00 | 477.30 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-04-04 10:30:00 | 477.30 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-04-04 12:45:00 | 483.40 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-04-04 13:15:00 | 483.90 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-04-04 14:15:00 | 483.85 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-04-04 15:00:00 | 483.00 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-04-16 12:00:00 | 501.20 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-04-17 09:15:00 | 501.90 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-04-17 12:30:00 | 500.65 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-04-17 13:30:00 | 502.00 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-04-23 11:15:00 | 508.20 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-04-23 13:00:00 | 507.85 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-04-24 10:00:00 | 511.85 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-05-07 11:00:00 | 524.15 | 2025-05-14 15:15:00 | 576.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-23 11:00:00 | 562.70 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-23 11:30:00 | 560.65 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-11 14:00:00 | 570.30 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-11 15:00:00 | 569.80 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-25 10:30:00 | 579.90 | 2025-07-01 15:15:00 | 588.00 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-03 09:15:00 | 620.15 | 2025-07-09 10:15:00 | 615.55 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-15 13:30:00 | 597.80 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-15 15:15:00 | 598.50 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-17 09:15:00 | 598.85 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-17 13:00:00 | 597.30 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-18 10:15:00 | 592.10 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-18 13:30:00 | 593.65 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-21 09:15:00 | 587.25 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-24 12:15:00 | 599.45 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-24 12:45:00 | 599.40 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-24 14:00:00 | 599.55 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-01 14:30:00 | 598.15 | 2025-08-04 09:15:00 | 586.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-08 11:45:00 | 588.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 15:15:00 | 587.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-18 09:15:00 | 612.40 | 2025-08-21 09:15:00 | 614.10 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 574.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 571.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 574.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.26% |
| BUY | retest2 | 2025-09-08 14:15:00 | 645.00 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-08 14:45:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-10 09:15:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-11 09:15:00 | 642.25 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-23 15:15:00 | 639.60 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-24 10:00:00 | 641.80 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-29 11:15:00 | 626.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-29 13:15:00 | 627.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-29 14:30:00 | 626.65 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-30 14:45:00 | 627.40 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-08 09:15:00 | 667.25 | 2025-10-14 15:15:00 | 685.00 | STOP_HIT | 1.00 | 2.66% |
| SELL | retest2 | 2025-10-30 10:15:00 | 693.95 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-10-30 13:45:00 | 694.75 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-31 09:15:00 | 678.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-03 10:45:00 | 691.20 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-03 13:15:00 | 681.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-04 09:30:00 | 680.10 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-07 10:45:00 | 694.15 | 2025-11-10 09:15:00 | 680.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-11-11 09:15:00 | 672.35 | 2025-11-12 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-13 14:45:00 | 675.30 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-14 09:45:00 | 676.80 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-14 10:30:00 | 676.05 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-14 12:45:00 | 676.40 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-17 09:15:00 | 673.35 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-11-17 11:15:00 | 674.65 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-17 12:45:00 | 675.90 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-12-01 13:30:00 | 666.85 | 2025-12-02 09:15:00 | 658.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 09:15:00 | 616.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 14:15:00 | 622.00 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest2 | 2025-12-29 13:00:00 | 613.30 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-29 13:45:00 | 614.00 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-03 10:45:00 | 563.50 | 2026-02-03 14:15:00 | 546.30 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-16 10:00:00 | 607.85 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2026-02-16 10:30:00 | 608.00 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2026-03-10 09:15:00 | 665.05 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-03-10 11:00:00 | 667.60 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-03-24 10:30:00 | 624.80 | 2026-03-24 11:15:00 | 638.95 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-30 12:15:00 | 653.00 | 2026-04-02 15:15:00 | 652.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 671.65 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-04-10 10:30:00 | 671.80 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-10 11:00:00 | 671.85 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-10 12:30:00 | 672.10 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-04-13 10:45:00 | 679.45 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-04-17 09:15:00 | 677.85 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-20 13:45:00 | 678.30 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.19% |
