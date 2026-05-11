# Shriram Finance Ltd. (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1003.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 233 |
| ALERT1 | 166 |
| ALERT2 | 162 |
| ALERT2_SKIP | 77 |
| ALERT3 | 471 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 168 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 184 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 130
- **Target hits / Stop hits / Partials:** 8 / 167 / 9
- **Avg / median % per leg:** 0.19% / -0.75%
- **Sum % (uncompounded):** 34.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 93 | 33 | 35.5% | 8 | 85 | 0 | 0.56% | 52.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 6 | 0 | -0.23% | -1.4% |
| BUY @ 3rd Alert (retest2) | 87 | 30 | 34.5% | 8 | 79 | 0 | 0.62% | 53.8% |
| SELL (all) | 91 | 21 | 23.1% | 0 | 82 | 9 | -0.19% | -17.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.33% | -0.3% |
| SELL @ 3rd Alert (retest2) | 90 | 21 | 23.3% | 0 | 81 | 9 | -0.19% | -17.1% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 7 | 0 | -0.25% | -1.7% |
| retest2 (combined) | 177 | 51 | 28.8% | 8 | 160 | 9 | 0.21% | 36.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 271.61 | 271.20 | 271.17 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 10:15:00 | 270.88 | 271.14 | 271.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 11:15:00 | 270.42 | 270.99 | 271.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 09:15:00 | 270.38 | 269.78 | 270.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 09:15:00 | 270.38 | 269.78 | 270.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 09:15:00 | 270.38 | 269.78 | 270.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-16 12:00:00 | 268.67 | 269.63 | 270.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 09:15:00 | 268.20 | 267.79 | 268.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 15:15:00 | 270.25 | 268.36 | 268.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 15:15:00 | 270.25 | 268.36 | 268.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 10:15:00 | 270.83 | 268.76 | 268.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 14:15:00 | 267.95 | 268.94 | 268.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 267.95 | 268.94 | 268.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 267.95 | 268.94 | 268.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 267.95 | 268.94 | 268.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 267.07 | 268.56 | 268.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 263.68 | 268.56 | 268.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 262.78 | 267.41 | 268.00 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 271.86 | 267.56 | 267.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 273.20 | 268.69 | 267.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 273.39 | 273.52 | 271.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 273.39 | 273.52 | 271.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 273.39 | 273.52 | 271.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:00:00 | 273.39 | 273.52 | 271.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 276.12 | 274.68 | 273.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:30:00 | 274.76 | 274.68 | 273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 274.24 | 274.59 | 273.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:45:00 | 275.08 | 274.59 | 273.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 274.69 | 274.50 | 273.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 13:30:00 | 275.50 | 274.72 | 273.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:00:00 | 275.60 | 274.72 | 273.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 14:15:00 | 280.17 | 281.55 | 281.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 280.17 | 281.55 | 281.72 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 10:15:00 | 283.35 | 281.80 | 281.78 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 281.40 | 281.74 | 281.76 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 14:15:00 | 282.18 | 281.83 | 281.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 09:15:00 | 282.76 | 282.04 | 281.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 11:15:00 | 281.40 | 282.23 | 282.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 11:15:00 | 281.40 | 282.23 | 282.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 281.40 | 282.23 | 282.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:00:00 | 281.40 | 282.23 | 282.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 280.93 | 281.97 | 281.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:45:00 | 280.73 | 281.97 | 281.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 281.96 | 281.98 | 281.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 14:45:00 | 281.86 | 281.98 | 281.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 15:15:00 | 281.20 | 281.82 | 281.87 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 09:15:00 | 283.29 | 282.11 | 282.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 288.00 | 283.29 | 282.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 12:15:00 | 283.00 | 283.40 | 282.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 12:15:00 | 283.00 | 283.40 | 282.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 283.00 | 283.40 | 282.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:45:00 | 282.89 | 283.40 | 282.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 283.00 | 283.32 | 282.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:00:00 | 283.00 | 283.32 | 282.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 282.07 | 283.07 | 282.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 282.07 | 283.07 | 282.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 283.98 | 283.25 | 282.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 284.29 | 283.25 | 282.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 287.17 | 284.04 | 283.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 12:30:00 | 287.80 | 285.06 | 284.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 14:30:00 | 287.71 | 286.41 | 284.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 11:15:00 | 287.60 | 287.96 | 286.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 12:15:00 | 282.87 | 285.87 | 286.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 12:15:00 | 282.87 | 285.87 | 286.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 13:15:00 | 282.82 | 285.26 | 285.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 13:15:00 | 281.28 | 281.27 | 282.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-16 14:00:00 | 281.28 | 281.27 | 282.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 13 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 294.44 | 283.69 | 283.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 12:15:00 | 304.92 | 295.40 | 290.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 332.54 | 339.86 | 331.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 332.54 | 339.86 | 331.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 332.54 | 339.86 | 331.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 332.54 | 339.86 | 331.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 330.80 | 338.05 | 331.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 11:00:00 | 330.80 | 338.05 | 331.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 329.15 | 336.27 | 330.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:00:00 | 329.15 | 336.27 | 330.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 334.33 | 335.88 | 331.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 14:00:00 | 335.89 | 333.60 | 332.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 13:45:00 | 337.24 | 336.65 | 334.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 09:15:00 | 349.87 | 355.34 | 355.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 349.87 | 355.34 | 355.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 345.13 | 353.30 | 354.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 14:15:00 | 344.59 | 344.28 | 347.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 353.38 | 346.21 | 347.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 353.38 | 346.21 | 347.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 353.38 | 346.21 | 347.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 352.40 | 347.45 | 348.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:45:00 | 354.00 | 347.45 | 348.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 353.60 | 348.68 | 348.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 357.62 | 352.20 | 350.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 358.91 | 360.05 | 356.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 358.91 | 360.05 | 356.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 358.19 | 359.68 | 357.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 15:00:00 | 358.19 | 359.68 | 357.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 355.99 | 358.71 | 357.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:00:00 | 355.99 | 358.71 | 357.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 355.59 | 358.09 | 357.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:45:00 | 355.58 | 358.09 | 357.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 356.70 | 357.47 | 357.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 357.58 | 357.47 | 357.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 356.74 | 357.32 | 357.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:15:00 | 355.49 | 357.32 | 357.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 10:15:00 | 354.61 | 356.78 | 356.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 11:15:00 | 353.19 | 356.06 | 356.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 15:15:00 | 355.99 | 354.77 | 355.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 15:15:00 | 355.99 | 354.77 | 355.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 355.99 | 354.77 | 355.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:15:00 | 356.15 | 354.77 | 355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 358.41 | 355.50 | 355.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:45:00 | 358.21 | 355.50 | 355.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 10:15:00 | 359.94 | 356.39 | 356.26 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 354.91 | 356.16 | 356.18 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 13:15:00 | 356.43 | 356.21 | 356.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 363.63 | 358.13 | 357.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 09:15:00 | 362.62 | 364.57 | 361.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 362.62 | 364.57 | 361.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 362.62 | 364.57 | 361.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 362.87 | 364.57 | 361.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 361.76 | 363.75 | 362.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:30:00 | 360.78 | 363.75 | 362.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 364.48 | 363.90 | 362.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 12:45:00 | 362.39 | 363.90 | 362.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 361.67 | 363.45 | 362.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 14:00:00 | 361.67 | 363.45 | 362.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 360.00 | 362.76 | 361.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 14:45:00 | 358.00 | 362.76 | 361.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 356.80 | 361.57 | 361.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 361.86 | 361.57 | 361.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 13:15:00 | 361.16 | 361.86 | 361.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 09:15:00 | 361.37 | 363.63 | 363.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 361.37 | 363.63 | 363.69 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 372.80 | 365.38 | 364.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 378.60 | 372.42 | 368.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 371.60 | 373.18 | 369.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 09:30:00 | 370.94 | 373.18 | 369.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 370.08 | 372.56 | 369.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:30:00 | 370.51 | 372.56 | 369.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 372.39 | 372.21 | 370.05 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 366.40 | 369.49 | 369.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 361.97 | 367.99 | 368.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 367.12 | 367.01 | 368.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 367.12 | 367.01 | 368.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 366.19 | 364.10 | 365.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:45:00 | 366.48 | 364.10 | 365.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 365.94 | 364.46 | 365.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 13:00:00 | 365.94 | 364.46 | 365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 367.75 | 365.17 | 365.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 15:00:00 | 367.75 | 365.17 | 365.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 366.92 | 365.52 | 365.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:15:00 | 369.43 | 365.52 | 365.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 368.85 | 366.18 | 366.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 370.47 | 368.03 | 367.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 369.71 | 370.25 | 368.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 369.71 | 370.25 | 368.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 369.80 | 370.16 | 368.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 369.56 | 370.16 | 368.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 372.01 | 370.92 | 369.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:30:00 | 369.31 | 370.92 | 369.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 370.17 | 371.01 | 369.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:30:00 | 369.33 | 371.01 | 369.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 370.50 | 370.91 | 369.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 371.19 | 370.91 | 369.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 368.80 | 370.49 | 369.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 12:45:00 | 368.80 | 370.49 | 369.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 370.00 | 370.39 | 369.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 13:45:00 | 368.63 | 370.39 | 369.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 369.95 | 370.30 | 369.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 369.95 | 370.30 | 369.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 370.55 | 370.35 | 369.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 09:30:00 | 372.29 | 371.10 | 370.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 10:30:00 | 372.27 | 371.00 | 370.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 09:15:00 | 367.69 | 369.69 | 369.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 367.69 | 369.69 | 369.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 12:15:00 | 364.00 | 367.47 | 368.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 363.15 | 361.45 | 364.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 14:00:00 | 363.15 | 361.45 | 364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 362.94 | 361.15 | 362.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 362.94 | 361.15 | 362.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 360.71 | 361.07 | 362.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 14:45:00 | 359.38 | 360.59 | 362.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 09:15:00 | 369.77 | 362.35 | 362.75 | SL hit (close>static) qty=1.00 sl=363.30 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 369.94 | 363.86 | 363.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 14:15:00 | 370.35 | 368.27 | 367.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 14:15:00 | 375.85 | 376.50 | 373.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 15:00:00 | 375.85 | 376.50 | 373.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 373.98 | 375.67 | 373.32 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 10:15:00 | 371.24 | 372.57 | 372.73 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 375.59 | 372.81 | 372.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 381.40 | 374.69 | 373.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 377.18 | 377.39 | 375.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 14:00:00 | 377.18 | 377.39 | 375.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 384.42 | 384.01 | 382.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:30:00 | 382.27 | 384.01 | 382.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 378.90 | 382.99 | 382.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 378.90 | 382.99 | 382.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 378.89 | 382.17 | 381.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:00:00 | 378.89 | 382.17 | 381.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 15:15:00 | 379.07 | 381.55 | 381.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 10:15:00 | 376.19 | 380.03 | 380.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 13:15:00 | 384.64 | 380.42 | 380.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 13:15:00 | 384.64 | 380.42 | 380.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 384.64 | 380.42 | 380.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:00:00 | 384.64 | 380.42 | 380.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 383.01 | 380.94 | 380.97 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 15:15:00 | 383.40 | 381.43 | 381.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 14:15:00 | 379.68 | 381.09 | 381.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 10:15:00 | 376.02 | 379.28 | 380.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 13:15:00 | 379.00 | 378.60 | 379.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-07 14:00:00 | 379.00 | 378.60 | 379.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 379.46 | 378.77 | 379.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 14:30:00 | 379.64 | 378.77 | 379.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 379.80 | 378.98 | 379.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:15:00 | 387.44 | 378.98 | 379.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 09:15:00 | 385.74 | 380.33 | 380.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 389.93 | 383.30 | 381.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 385.82 | 389.50 | 387.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 385.82 | 389.50 | 387.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 385.82 | 389.50 | 387.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:00:00 | 385.82 | 389.50 | 387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 386.25 | 388.85 | 387.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 11:30:00 | 387.68 | 388.00 | 387.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 12:15:00 | 380.95 | 386.59 | 386.53 | SL hit (close<static) qty=1.00 sl=383.12 alert=retest2 |

### Cycle 32 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 383.47 | 385.96 | 386.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 15:15:00 | 378.80 | 383.96 | 385.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 382.51 | 381.44 | 383.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 382.51 | 381.44 | 383.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 382.51 | 381.44 | 383.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:30:00 | 382.75 | 381.44 | 383.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 383.19 | 381.77 | 382.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 384.98 | 381.77 | 382.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 383.91 | 382.20 | 382.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 383.73 | 382.20 | 382.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 383.79 | 382.52 | 383.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:15:00 | 382.63 | 382.52 | 383.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:30:00 | 382.80 | 382.83 | 383.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 15:15:00 | 382.41 | 382.94 | 383.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:15:00 | 382.85 | 383.04 | 383.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 383.35 | 382.91 | 383.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 13:45:00 | 383.63 | 382.91 | 383.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-15 14:15:00 | 384.19 | 383.17 | 383.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 14:15:00 | 384.19 | 383.17 | 383.13 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 381.94 | 383.01 | 383.10 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 387.35 | 383.49 | 383.22 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 381.44 | 383.71 | 383.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 13:15:00 | 378.34 | 380.43 | 381.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 381.17 | 377.79 | 379.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 381.17 | 377.79 | 379.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 381.17 | 377.79 | 379.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 12:30:00 | 378.50 | 377.79 | 379.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 381.45 | 378.52 | 379.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 382.94 | 378.52 | 379.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 388.94 | 380.61 | 380.48 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 378.88 | 380.96 | 381.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 15:15:00 | 372.43 | 376.56 | 378.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 379.03 | 377.06 | 378.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 379.03 | 377.06 | 378.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 379.03 | 377.06 | 378.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 379.03 | 377.06 | 378.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 381.70 | 377.99 | 378.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 11:00:00 | 381.70 | 377.99 | 378.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 384.95 | 379.38 | 379.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 387.26 | 383.40 | 381.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 11:15:00 | 382.59 | 383.24 | 381.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-03 12:00:00 | 382.59 | 383.24 | 381.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 383.29 | 383.25 | 381.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:45:00 | 381.97 | 383.25 | 381.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 375.21 | 382.24 | 381.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 375.21 | 382.24 | 381.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 372.00 | 380.19 | 380.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 371.00 | 378.35 | 380.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 369.91 | 368.98 | 372.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:45:00 | 371.58 | 368.98 | 372.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 370.88 | 369.36 | 371.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:30:00 | 372.39 | 369.36 | 371.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 369.63 | 369.16 | 370.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:30:00 | 370.50 | 369.16 | 370.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 365.35 | 364.96 | 367.28 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 376.98 | 369.11 | 368.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 378.00 | 370.89 | 369.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 11:15:00 | 378.86 | 378.93 | 376.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 12:00:00 | 378.86 | 378.93 | 376.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 375.86 | 378.05 | 376.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 15:00:00 | 375.86 | 378.05 | 376.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 375.99 | 377.64 | 376.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 375.98 | 377.64 | 376.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 376.01 | 376.13 | 376.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:45:00 | 376.43 | 376.13 | 376.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 374.60 | 375.82 | 375.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 10:15:00 | 374.53 | 375.58 | 375.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 11:15:00 | 375.65 | 375.60 | 375.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 11:15:00 | 375.65 | 375.60 | 375.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 375.65 | 375.60 | 375.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 12:00:00 | 375.65 | 375.60 | 375.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 12:15:00 | 375.00 | 375.48 | 375.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 13:30:00 | 374.70 | 375.18 | 375.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:30:00 | 374.82 | 375.38 | 375.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 14:15:00 | 377.64 | 375.83 | 375.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 377.64 | 375.83 | 375.63 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 372.26 | 375.03 | 375.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 369.80 | 373.00 | 374.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 374.08 | 372.76 | 373.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 374.08 | 372.76 | 373.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 374.08 | 372.76 | 373.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:00:00 | 374.08 | 372.76 | 373.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 374.68 | 373.14 | 373.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:00:00 | 374.68 | 373.14 | 373.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 377.40 | 373.99 | 374.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 377.40 | 373.99 | 374.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 15:15:00 | 376.34 | 374.71 | 374.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 09:15:00 | 379.61 | 376.18 | 375.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 14:15:00 | 376.21 | 377.45 | 376.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 14:15:00 | 376.21 | 377.45 | 376.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 14:15:00 | 376.21 | 377.45 | 376.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 15:00:00 | 376.21 | 377.45 | 376.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 375.00 | 376.96 | 376.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:15:00 | 370.12 | 376.96 | 376.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 09:15:00 | 369.49 | 375.47 | 375.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 10:15:00 | 366.45 | 373.66 | 374.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 359.24 | 358.76 | 364.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 15:00:00 | 359.24 | 358.76 | 364.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 389.14 | 365.03 | 365.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 390.41 | 365.03 | 365.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 395.47 | 371.12 | 368.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 398.88 | 389.54 | 384.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 12:15:00 | 399.10 | 399.84 | 395.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 13:00:00 | 399.10 | 399.84 | 395.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 397.81 | 398.98 | 396.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:30:00 | 395.51 | 398.98 | 396.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 394.00 | 397.98 | 396.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 394.00 | 397.98 | 396.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 392.38 | 396.86 | 395.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:30:00 | 391.95 | 396.86 | 395.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 397.32 | 397.04 | 396.13 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 393.33 | 395.59 | 395.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 388.40 | 392.09 | 393.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 12:15:00 | 391.16 | 390.13 | 392.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 12:15:00 | 391.16 | 390.13 | 392.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 391.16 | 390.13 | 392.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:30:00 | 388.74 | 390.42 | 391.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 13:15:00 | 393.55 | 391.22 | 391.51 | SL hit (close>static) qty=1.00 sl=393.19 alert=retest2 |

### Cycle 49 — BUY (started 2023-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 15:15:00 | 394.00 | 392.02 | 391.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 398.44 | 393.31 | 392.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 09:15:00 | 400.50 | 404.77 | 402.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 400.50 | 404.77 | 402.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 400.50 | 404.77 | 402.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:30:00 | 398.30 | 404.77 | 402.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 399.94 | 403.81 | 402.03 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 14:15:00 | 399.40 | 400.93 | 401.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 15:15:00 | 397.61 | 400.26 | 400.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 15:15:00 | 398.76 | 398.72 | 399.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 09:15:00 | 399.05 | 398.72 | 399.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 397.96 | 398.57 | 399.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 10:30:00 | 397.20 | 398.13 | 399.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 12:15:00 | 403.20 | 399.31 | 399.51 | SL hit (close>static) qty=1.00 sl=401.39 alert=retest2 |

### Cycle 51 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 402.16 | 399.88 | 399.75 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 398.45 | 399.71 | 399.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 396.89 | 399.06 | 399.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 397.77 | 396.64 | 397.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 397.77 | 396.64 | 397.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 397.77 | 396.64 | 397.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 11:00:00 | 394.87 | 396.29 | 397.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 13:45:00 | 395.18 | 395.90 | 397.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:00:00 | 394.97 | 395.71 | 396.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 12:15:00 | 394.91 | 392.90 | 392.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 394.91 | 392.90 | 392.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 395.80 | 393.48 | 393.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 400.83 | 401.13 | 399.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 400.83 | 401.13 | 399.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 410.00 | 411.71 | 409.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:15:00 | 409.38 | 411.71 | 409.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 411.19 | 411.61 | 409.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 15:15:00 | 412.40 | 411.53 | 410.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:45:00 | 412.25 | 411.59 | 410.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 11:15:00 | 408.60 | 410.77 | 410.18 | SL hit (close<static) qty=1.00 sl=409.00 alert=retest2 |

### Cycle 54 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 407.19 | 409.38 | 409.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 11:15:00 | 405.73 | 408.03 | 408.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 15:15:00 | 406.78 | 406.72 | 407.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 09:15:00 | 407.33 | 406.72 | 407.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 410.93 | 407.57 | 408.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:00:00 | 410.93 | 407.57 | 408.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 411.97 | 408.45 | 408.47 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 409.63 | 408.68 | 408.58 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 09:15:00 | 405.04 | 408.29 | 408.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 10:15:00 | 402.00 | 407.03 | 407.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 15:15:00 | 399.99 | 399.84 | 402.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-14 09:15:00 | 406.63 | 399.84 | 402.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 416.65 | 403.20 | 403.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 416.65 | 403.20 | 403.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 417.01 | 405.96 | 404.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 422.40 | 409.25 | 406.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 12:15:00 | 417.77 | 418.43 | 414.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 13:00:00 | 417.77 | 418.43 | 414.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 414.85 | 417.62 | 414.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 414.85 | 417.62 | 414.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 415.00 | 417.09 | 414.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 413.46 | 417.09 | 414.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 413.93 | 416.46 | 414.43 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 409.59 | 412.94 | 413.33 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 419.03 | 413.56 | 413.15 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 400.06 | 410.85 | 412.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 398.80 | 408.44 | 411.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 412.39 | 406.20 | 408.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 412.39 | 406.20 | 408.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 412.39 | 406.20 | 408.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 412.39 | 406.20 | 408.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 410.64 | 407.09 | 408.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 409.04 | 407.95 | 408.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 15:15:00 | 411.12 | 409.23 | 408.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 411.12 | 409.23 | 408.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 412.99 | 409.98 | 409.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 10:15:00 | 409.54 | 409.89 | 409.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 10:15:00 | 409.54 | 409.89 | 409.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 409.54 | 409.89 | 409.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:45:00 | 406.95 | 409.89 | 409.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 11:15:00 | 408.76 | 409.67 | 409.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 12:00:00 | 408.76 | 409.67 | 409.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 406.94 | 409.12 | 409.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 13:00:00 | 406.94 | 409.12 | 409.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 13:15:00 | 409.01 | 409.10 | 409.08 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2023-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 14:15:00 | 407.54 | 408.79 | 408.94 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 411.50 | 409.39 | 409.16 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 13:15:00 | 408.00 | 409.04 | 409.16 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 11:15:00 | 411.90 | 409.64 | 409.39 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 406.52 | 409.32 | 409.64 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 10:15:00 | 416.65 | 409.69 | 409.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 13:15:00 | 418.83 | 413.23 | 411.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 425.72 | 426.31 | 421.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 10:45:00 | 424.99 | 426.31 | 421.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 433.74 | 436.89 | 434.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:30:00 | 430.68 | 436.89 | 434.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 434.25 | 436.36 | 434.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 428.14 | 436.36 | 434.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 430.10 | 435.11 | 434.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 430.66 | 435.11 | 434.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 428.80 | 433.85 | 433.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 428.80 | 433.85 | 433.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 429.40 | 432.96 | 433.22 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 450.22 | 435.26 | 433.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 10:15:00 | 458.68 | 451.55 | 444.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 459.84 | 462.57 | 458.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 459.84 | 462.57 | 458.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 458.49 | 461.49 | 459.09 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 447.53 | 456.41 | 457.43 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 458.86 | 456.46 | 456.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 12:15:00 | 460.93 | 457.35 | 456.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 15:15:00 | 458.22 | 458.22 | 457.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:30:00 | 463.00 | 458.62 | 457.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 11:30:00 | 461.91 | 459.45 | 458.17 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 460.62 | 459.68 | 458.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:30:00 | 458.40 | 459.68 | 458.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 458.77 | 459.77 | 458.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-20 15:15:00 | 458.77 | 459.77 | 458.78 | SL hit (close<ema400) qty=1.00 sl=458.78 alert=retest1 |

### Cycle 72 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 455.95 | 458.00 | 458.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 450.45 | 456.00 | 457.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 457.02 | 456.08 | 456.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 12:15:00 | 457.02 | 456.08 | 456.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 457.02 | 456.08 | 456.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:45:00 | 457.54 | 456.08 | 456.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 457.20 | 456.31 | 456.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:45:00 | 457.11 | 456.31 | 456.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 459.98 | 457.04 | 457.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 459.98 | 457.04 | 457.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 459.31 | 457.49 | 457.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 465.63 | 459.12 | 458.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 484.82 | 487.88 | 481.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 484.82 | 487.88 | 481.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 484.82 | 487.88 | 481.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 484.82 | 487.88 | 481.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 488.38 | 487.41 | 481.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 11:30:00 | 482.85 | 487.41 | 481.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 481.31 | 488.79 | 485.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:15:00 | 479.56 | 488.79 | 485.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 481.16 | 487.27 | 484.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 12:00:00 | 482.77 | 486.37 | 484.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:00:00 | 483.47 | 485.79 | 484.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:45:00 | 482.69 | 485.03 | 484.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 15:15:00 | 478.00 | 482.73 | 483.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 478.00 | 482.73 | 483.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 475.51 | 479.88 | 481.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 12:15:00 | 475.25 | 475.13 | 477.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 13:00:00 | 475.25 | 475.13 | 477.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 476.78 | 475.46 | 477.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:00:00 | 476.78 | 475.46 | 477.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 479.16 | 476.20 | 477.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 15:00:00 | 479.16 | 476.20 | 477.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 479.59 | 476.88 | 477.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:15:00 | 479.98 | 476.88 | 477.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 485.44 | 478.59 | 478.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 10:15:00 | 489.03 | 480.68 | 479.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 478.00 | 481.45 | 480.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 478.00 | 481.45 | 480.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 478.00 | 481.45 | 480.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:30:00 | 481.03 | 481.45 | 480.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 475.33 | 480.23 | 480.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 475.33 | 480.23 | 480.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 11:15:00 | 474.02 | 478.99 | 479.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 13:15:00 | 470.51 | 476.62 | 478.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 460.00 | 458.26 | 463.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:00:00 | 460.00 | 458.26 | 463.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 463.43 | 459.30 | 463.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 463.43 | 459.30 | 463.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 461.73 | 459.78 | 463.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:30:00 | 463.72 | 459.78 | 463.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 461.57 | 460.14 | 463.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 461.57 | 460.14 | 463.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 463.78 | 460.87 | 463.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:30:00 | 464.45 | 460.87 | 463.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 466.79 | 462.05 | 463.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 466.79 | 462.05 | 463.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 467.00 | 463.04 | 463.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 462.51 | 463.04 | 463.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 09:15:00 | 469.63 | 464.36 | 464.37 | SL hit (close>static) qty=1.00 sl=467.67 alert=retest2 |

### Cycle 77 — BUY (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 10:15:00 | 471.46 | 465.78 | 465.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 11:15:00 | 473.50 | 467.32 | 465.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 15:15:00 | 487.29 | 488.14 | 483.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 09:15:00 | 486.62 | 488.14 | 483.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 486.79 | 487.43 | 483.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:45:00 | 484.91 | 487.43 | 483.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 488.64 | 488.39 | 486.07 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 14:15:00 | 483.56 | 484.78 | 484.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 09:15:00 | 478.04 | 483.18 | 484.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 479.47 | 476.71 | 478.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 479.47 | 476.71 | 478.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 479.47 | 476.71 | 478.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 479.47 | 476.71 | 478.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 478.76 | 477.12 | 478.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 480.00 | 477.12 | 478.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 484.18 | 479.01 | 479.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:00:00 | 484.18 | 479.01 | 479.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 489.92 | 481.19 | 480.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 500.68 | 488.73 | 484.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 09:15:00 | 490.34 | 493.93 | 489.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 490.34 | 493.93 | 489.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 490.34 | 493.93 | 489.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 490.34 | 493.93 | 489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 487.42 | 492.63 | 489.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:45:00 | 486.87 | 492.63 | 489.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 488.85 | 491.87 | 489.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 492.41 | 490.78 | 489.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 462.90 | 485.20 | 486.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 462.90 | 485.20 | 486.97 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 487.59 | 482.00 | 481.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 490.50 | 483.70 | 482.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 489.31 | 489.71 | 487.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 10:15:00 | 485.44 | 488.85 | 487.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 485.44 | 488.85 | 487.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 485.44 | 488.85 | 487.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 484.48 | 487.98 | 486.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:45:00 | 484.90 | 487.98 | 486.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 488.73 | 487.50 | 486.81 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 481.91 | 486.86 | 487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 476.76 | 484.04 | 485.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 479.06 | 478.36 | 481.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 479.06 | 478.36 | 481.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 479.06 | 478.36 | 481.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 478.37 | 478.36 | 481.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 484.20 | 479.53 | 481.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 484.20 | 479.53 | 481.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 485.80 | 480.78 | 482.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:45:00 | 485.57 | 480.78 | 482.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 13:15:00 | 489.89 | 483.80 | 483.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 10:15:00 | 493.71 | 487.97 | 485.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 492.89 | 493.30 | 489.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 492.89 | 493.30 | 489.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 492.89 | 493.30 | 489.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:30:00 | 491.70 | 493.30 | 489.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 490.40 | 492.72 | 490.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 488.80 | 492.72 | 490.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 487.44 | 491.67 | 489.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:45:00 | 487.78 | 491.67 | 489.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 489.82 | 491.30 | 489.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 13:15:00 | 491.07 | 491.30 | 489.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 14:15:00 | 484.46 | 489.49 | 489.19 | SL hit (close<static) qty=1.00 sl=485.20 alert=retest2 |

### Cycle 84 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 485.80 | 488.75 | 488.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 474.98 | 486.00 | 487.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 15:15:00 | 463.19 | 461.90 | 468.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 09:30:00 | 459.79 | 461.16 | 467.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 461.30 | 456.15 | 461.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-18 09:15:00 | 461.30 | 456.15 | 461.26 | SL hit (close>ema400) qty=1.00 sl=461.26 alert=retest1 |

### Cycle 85 — BUY (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 12:15:00 | 463.38 | 462.25 | 462.19 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 13:15:00 | 459.85 | 461.77 | 461.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 457.19 | 460.85 | 461.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 460.40 | 460.26 | 461.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 460.40 | 460.26 | 461.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 460.40 | 460.26 | 461.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:45:00 | 462.22 | 460.26 | 461.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 462.00 | 460.61 | 461.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 464.04 | 460.61 | 461.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 461.36 | 460.76 | 461.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 12:00:00 | 461.36 | 460.76 | 461.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 462.98 | 461.20 | 461.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 462.98 | 461.20 | 461.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 463.56 | 461.68 | 461.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 469.61 | 463.34 | 462.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 10:15:00 | 460.84 | 462.84 | 462.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 10:15:00 | 460.84 | 462.84 | 462.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 460.84 | 462.84 | 462.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:45:00 | 460.02 | 462.84 | 462.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 458.92 | 462.06 | 461.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-21 11:30:00 | 459.09 | 462.06 | 461.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 463.60 | 462.24 | 462.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 465.42 | 462.24 | 462.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 14:15:00 | 464.96 | 465.44 | 464.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 10:15:00 | 463.94 | 464.97 | 464.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:00:00 | 463.99 | 464.78 | 464.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 466.15 | 464.91 | 464.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:30:00 | 462.93 | 464.91 | 464.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 473.26 | 474.59 | 471.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 481.20 | 472.98 | 472.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 11:00:00 | 481.64 | 475.10 | 473.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-03 09:15:00 | 511.96 | 492.44 | 486.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 506.34 | 507.87 | 507.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 14:15:00 | 505.24 | 507.35 | 507.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 10:15:00 | 486.87 | 483.54 | 487.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 11:00:00 | 486.87 | 483.54 | 487.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 485.62 | 483.95 | 487.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:30:00 | 487.21 | 483.95 | 487.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 487.00 | 484.56 | 487.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:00:00 | 487.00 | 484.56 | 487.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 483.00 | 484.25 | 486.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:45:00 | 477.02 | 482.07 | 485.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 485.72 | 480.97 | 480.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 485.72 | 480.97 | 480.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 490.57 | 483.59 | 481.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 485.13 | 485.84 | 483.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 485.13 | 485.84 | 483.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 485.13 | 485.84 | 483.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 485.13 | 485.84 | 483.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 488.48 | 486.27 | 484.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 11:45:00 | 490.65 | 487.56 | 485.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:45:00 | 492.93 | 491.09 | 488.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 12:15:00 | 472.45 | 488.67 | 489.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 12:15:00 | 472.45 | 488.67 | 489.38 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-04-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 15:15:00 | 501.80 | 490.65 | 489.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 11:15:00 | 509.50 | 499.81 | 495.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 14:15:00 | 518.17 | 518.61 | 513.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 14:45:00 | 516.80 | 518.61 | 513.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 502.68 | 515.11 | 512.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 502.68 | 515.11 | 512.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 510.04 | 514.10 | 512.25 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 15:15:00 | 508.01 | 510.99 | 511.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 505.44 | 509.88 | 510.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 466.68 | 464.20 | 470.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 466.68 | 464.20 | 470.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 466.68 | 464.20 | 470.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 464.49 | 465.87 | 468.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 10:00:00 | 462.81 | 465.25 | 468.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 13:15:00 | 464.59 | 465.53 | 467.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 13:45:00 | 464.57 | 465.79 | 467.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 458.61 | 463.84 | 466.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-17 11:15:00 | 469.17 | 465.17 | 465.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 469.17 | 465.17 | 465.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 12:15:00 | 470.39 | 466.22 | 465.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 469.19 | 473.18 | 471.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 469.19 | 473.18 | 471.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 469.19 | 473.18 | 471.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 470.70 | 473.18 | 471.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 467.84 | 472.11 | 471.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 468.00 | 472.11 | 471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 467.68 | 470.42 | 470.68 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 474.69 | 471.27 | 470.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 477.34 | 472.86 | 471.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 474.20 | 475.01 | 473.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 474.20 | 475.01 | 473.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 474.20 | 475.01 | 473.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 474.20 | 475.01 | 473.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 478.72 | 475.53 | 473.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 480.40 | 475.53 | 473.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 479.37 | 479.15 | 476.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:30:00 | 480.00 | 478.99 | 477.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 478.85 | 478.59 | 477.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 477.71 | 478.41 | 477.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:30:00 | 477.39 | 478.41 | 477.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 476.36 | 478.00 | 477.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 475.70 | 478.00 | 477.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 476.04 | 477.61 | 477.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:45:00 | 476.20 | 477.61 | 477.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 479.56 | 478.00 | 477.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 476.40 | 477.44 | 477.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 11:15:00 | 476.40 | 477.44 | 477.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 466.05 | 474.48 | 476.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 469.87 | 465.69 | 469.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 469.87 | 465.69 | 469.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 469.87 | 465.69 | 469.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 469.87 | 465.69 | 469.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 466.78 | 465.91 | 469.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 466.26 | 467.14 | 469.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 477.00 | 469.66 | 469.94 | SL hit (close>static) qty=1.00 sl=471.15 alert=retest2 |

### Cycle 97 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 501.84 | 476.10 | 472.84 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 452.56 | 479.89 | 480.60 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 493.40 | 474.39 | 473.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 503.64 | 480.24 | 475.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 501.00 | 501.25 | 496.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 499.20 | 501.25 | 496.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 557.63 | 558.80 | 553.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:45:00 | 564.96 | 560.46 | 554.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 563.92 | 561.81 | 557.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 565.11 | 562.05 | 557.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 586.92 | 589.42 | 589.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 13:15:00 | 586.92 | 589.42 | 589.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 580.70 | 587.67 | 588.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 15:15:00 | 586.00 | 585.39 | 586.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:15:00 | 581.15 | 585.39 | 586.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 574.84 | 583.28 | 585.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:15:00 | 570.89 | 581.43 | 584.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 13:45:00 | 571.75 | 572.30 | 575.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 14:15:00 | 571.87 | 572.30 | 575.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 569.24 | 572.10 | 574.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 570.30 | 567.75 | 570.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 569.95 | 567.75 | 570.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 570.00 | 568.20 | 570.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:30:00 | 570.77 | 568.20 | 570.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 573.42 | 569.24 | 570.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 573.42 | 569.24 | 570.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 571.03 | 569.60 | 570.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 564.95 | 569.60 | 570.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 542.35 | 554.97 | 559.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 543.16 | 554.97 | 559.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 543.28 | 554.97 | 559.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-11 15:15:00 | 549.03 | 548.58 | 552.17 | SL hit (close>ema200) qty=0.50 sl=548.58 alert=retest2 |

### Cycle 101 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 564.52 | 554.59 | 554.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 570.55 | 561.42 | 558.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 567.80 | 569.45 | 564.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:45:00 | 569.41 | 569.45 | 564.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 561.48 | 567.37 | 565.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 561.48 | 567.37 | 565.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 564.00 | 566.69 | 565.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 558.23 | 566.69 | 565.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 565.68 | 566.49 | 565.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 569.56 | 565.90 | 565.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 569.00 | 568.52 | 566.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 565.13 | 566.02 | 566.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 565.13 | 566.02 | 566.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 559.96 | 564.31 | 565.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 565.40 | 564.35 | 565.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 565.40 | 564.35 | 565.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 565.40 | 564.35 | 565.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 565.40 | 564.35 | 565.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 567.19 | 564.92 | 565.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 567.19 | 564.92 | 565.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 567.32 | 565.40 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 568.58 | 565.40 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 565.97 | 565.51 | 565.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 565.30 | 565.47 | 565.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 15:15:00 | 566.12 | 565.60 | 565.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 566.12 | 565.60 | 565.57 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 550.73 | 562.63 | 564.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 547.16 | 557.86 | 561.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 549.64 | 538.91 | 543.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 549.64 | 538.91 | 543.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 549.64 | 538.91 | 543.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 547.28 | 538.91 | 543.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 552.40 | 541.61 | 543.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 552.40 | 541.61 | 543.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 553.00 | 545.57 | 545.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 582.04 | 552.87 | 548.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 580.65 | 582.22 | 571.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 09:30:00 | 584.10 | 582.22 | 571.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 591.90 | 592.04 | 587.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 587.15 | 592.04 | 587.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 583.70 | 594.55 | 591.90 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 570.01 | 589.65 | 589.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 10:15:00 | 569.58 | 578.39 | 583.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 581.59 | 573.92 | 578.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 581.59 | 573.92 | 578.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 581.59 | 573.92 | 578.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 581.59 | 573.92 | 578.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 579.96 | 575.13 | 578.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 584.18 | 575.13 | 578.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 577.39 | 575.78 | 578.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 577.93 | 575.78 | 578.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 581.80 | 576.98 | 578.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 581.80 | 576.98 | 578.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 585.60 | 578.70 | 579.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 585.60 | 578.70 | 579.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 584.60 | 579.88 | 579.60 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 576.92 | 579.29 | 579.36 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 581.00 | 579.47 | 579.42 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 578.42 | 579.26 | 579.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 574.29 | 578.06 | 578.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 585.44 | 578.81 | 578.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 585.44 | 578.81 | 578.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 585.44 | 578.81 | 578.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:45:00 | 586.53 | 578.81 | 578.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 586.77 | 580.40 | 579.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 11:15:00 | 590.60 | 582.44 | 580.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 587.48 | 593.04 | 589.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 587.48 | 593.04 | 589.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 587.48 | 593.04 | 589.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 589.01 | 593.04 | 589.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 585.86 | 591.60 | 589.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 587.21 | 591.60 | 589.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 576.25 | 585.88 | 587.00 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 590.27 | 584.49 | 584.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 597.28 | 587.80 | 585.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 627.49 | 629.04 | 620.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 627.49 | 629.04 | 620.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 626.30 | 629.07 | 625.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 626.30 | 629.07 | 625.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 626.58 | 628.57 | 626.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:45:00 | 626.65 | 628.57 | 626.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 627.44 | 628.35 | 626.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 627.38 | 628.35 | 626.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 625.44 | 627.77 | 626.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 625.44 | 627.77 | 626.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 625.00 | 627.21 | 626.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 625.00 | 627.21 | 626.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 625.60 | 626.47 | 625.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 624.36 | 626.47 | 625.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 630.27 | 630.67 | 628.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 624.46 | 630.67 | 628.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 631.94 | 630.92 | 629.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 627.66 | 630.92 | 629.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 638.35 | 641.28 | 638.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 645.60 | 641.28 | 638.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 650.56 | 643.14 | 639.57 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 636.80 | 639.52 | 639.65 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 641.49 | 639.92 | 639.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 650.47 | 642.36 | 640.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 14:15:00 | 645.67 | 645.70 | 643.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:45:00 | 644.68 | 645.70 | 643.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 648.71 | 646.22 | 644.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:45:00 | 655.54 | 646.23 | 644.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:00:00 | 654.71 | 648.85 | 646.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 653.35 | 649.56 | 646.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 657.54 | 648.56 | 647.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 649.72 | 650.47 | 648.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 649.72 | 650.47 | 648.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 649.12 | 650.20 | 648.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 649.12 | 650.20 | 648.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 649.09 | 649.98 | 648.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 15:15:00 | 649.99 | 649.98 | 648.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 649.89 | 649.35 | 648.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:45:00 | 650.32 | 649.04 | 648.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:15:00 | 649.61 | 649.04 | 648.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 652.07 | 649.64 | 648.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-06 15:15:00 | 647.60 | 648.48 | 648.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 647.60 | 648.48 | 648.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 646.06 | 648.00 | 648.34 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 10:15:00 | 652.06 | 648.81 | 648.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 11:15:00 | 657.24 | 650.50 | 649.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 650.62 | 656.16 | 653.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 650.62 | 656.16 | 653.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 650.62 | 656.16 | 653.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 650.62 | 656.16 | 653.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 650.11 | 654.95 | 653.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 650.08 | 654.95 | 653.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 653.00 | 653.70 | 652.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 651.57 | 653.70 | 652.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 651.76 | 653.31 | 652.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 651.76 | 653.31 | 652.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 651.73 | 652.99 | 652.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:30:00 | 654.59 | 652.80 | 652.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 654.47 | 653.47 | 652.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:00:00 | 655.85 | 654.77 | 653.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-19 09:15:00 | 720.05 | 707.06 | 696.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 703.60 | 706.71 | 706.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 697.78 | 704.92 | 706.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 12:15:00 | 701.57 | 701.21 | 703.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:00:00 | 701.57 | 701.21 | 703.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 705.54 | 702.06 | 703.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 705.54 | 702.06 | 703.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 706.05 | 702.86 | 703.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 708.51 | 702.86 | 703.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 709.75 | 705.03 | 704.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 711.08 | 706.85 | 705.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 723.09 | 723.52 | 718.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 11:00:00 | 723.09 | 723.52 | 718.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 720.16 | 722.38 | 718.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 722.79 | 719.67 | 718.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 12:15:00 | 716.71 | 718.97 | 718.47 | SL hit (close<static) qty=1.00 sl=718.16 alert=retest2 |

### Cycle 120 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 713.82 | 717.43 | 717.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 694.56 | 712.24 | 715.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 663.29 | 662.10 | 671.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 665.64 | 662.10 | 671.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 667.35 | 663.54 | 670.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 667.35 | 663.54 | 670.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 665.99 | 664.03 | 670.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 665.99 | 664.03 | 670.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 667.00 | 664.63 | 670.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 668.50 | 664.63 | 670.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 691.41 | 670.31 | 671.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 691.41 | 670.31 | 671.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 692.16 | 674.68 | 673.21 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 670.24 | 674.11 | 674.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 666.99 | 670.07 | 672.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 668.53 | 667.36 | 669.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 668.53 | 667.36 | 669.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 668.53 | 667.36 | 669.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:45:00 | 668.45 | 667.36 | 669.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 669.60 | 667.81 | 669.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 677.60 | 667.81 | 669.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 678.61 | 669.97 | 670.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 678.61 | 669.97 | 670.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 679.60 | 671.89 | 671.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 15:15:00 | 682.37 | 679.27 | 676.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 677.58 | 678.93 | 676.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 677.58 | 678.93 | 676.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 677.58 | 678.93 | 676.75 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 665.55 | 675.34 | 675.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 657.20 | 670.06 | 673.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 660.60 | 657.41 | 663.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 660.60 | 657.41 | 663.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 657.80 | 657.48 | 663.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 661.40 | 657.48 | 663.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 672.59 | 660.51 | 664.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 672.59 | 660.51 | 664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 668.50 | 662.10 | 664.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 665.11 | 663.79 | 664.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 664.13 | 662.88 | 663.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 668.50 | 664.00 | 663.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 10:15:00 | 668.50 | 664.00 | 663.91 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 660.60 | 663.71 | 663.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 649.42 | 660.26 | 662.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 645.77 | 645.15 | 650.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 12:15:00 | 646.78 | 645.15 | 650.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 650.05 | 646.62 | 650.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:00:00 | 650.05 | 646.62 | 650.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 648.86 | 647.07 | 650.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 648.50 | 647.07 | 650.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 651.17 | 647.89 | 650.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 641.96 | 647.89 | 650.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 13:15:00 | 609.86 | 628.18 | 638.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 649.89 | 630.50 | 636.92 | SL hit (close>ema200) qty=0.50 sl=630.50 alert=retest2 |

### Cycle 127 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 661.65 | 641.29 | 640.99 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 638.03 | 643.46 | 644.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 636.74 | 642.12 | 643.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 628.68 | 628.25 | 634.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 628.68 | 628.25 | 634.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 632.21 | 628.39 | 631.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 632.21 | 628.39 | 631.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 629.69 | 628.65 | 631.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 628.40 | 628.65 | 631.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 633.19 | 628.88 | 628.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 633.19 | 628.88 | 628.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 637.54 | 632.26 | 630.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 622.69 | 630.91 | 630.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 622.69 | 630.91 | 630.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 622.69 | 630.91 | 630.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 622.69 | 630.91 | 630.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 619.97 | 628.73 | 629.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 617.16 | 626.41 | 628.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 607.72 | 605.65 | 612.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 607.72 | 605.65 | 612.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 570.09 | 565.98 | 571.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 570.09 | 565.98 | 571.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 569.73 | 566.73 | 571.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:45:00 | 566.77 | 568.14 | 571.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:00:00 | 566.78 | 567.86 | 570.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 565.31 | 566.47 | 569.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 14:15:00 | 570.83 | 565.95 | 565.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 570.83 | 565.95 | 565.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 590.84 | 571.32 | 568.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 601.42 | 604.53 | 599.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 601.07 | 604.53 | 599.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 602.75 | 604.17 | 599.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 599.62 | 604.17 | 599.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 601.86 | 603.71 | 599.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 600.45 | 603.71 | 599.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 601.00 | 604.74 | 601.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 601.00 | 604.74 | 601.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 602.06 | 604.20 | 601.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:15:00 | 604.60 | 603.80 | 601.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 606.08 | 604.25 | 602.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 611.16 | 604.15 | 602.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 621.14 | 624.10 | 624.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 621.14 | 624.10 | 624.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 620.61 | 622.86 | 623.63 | Break + close below crossover candle low |

### Cycle 133 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 632.80 | 624.55 | 624.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 639.59 | 634.48 | 630.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 630.60 | 644.70 | 642.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 630.60 | 644.70 | 642.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 630.60 | 644.70 | 642.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 630.60 | 644.70 | 642.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 625.25 | 640.81 | 640.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 626.05 | 640.81 | 640.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 629.15 | 638.48 | 639.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 606.43 | 625.26 | 630.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 587.15 | 586.53 | 595.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 14:00:00 | 587.15 | 586.53 | 595.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 588.43 | 582.33 | 587.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 588.43 | 582.33 | 587.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 586.96 | 583.25 | 587.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 583.56 | 583.25 | 587.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 582.56 | 580.17 | 580.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 586.56 | 581.44 | 581.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 586.56 | 581.44 | 581.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 590.49 | 584.71 | 583.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 582.84 | 584.73 | 583.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 582.84 | 584.73 | 583.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 582.84 | 584.73 | 583.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 582.84 | 584.73 | 583.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 584.68 | 584.72 | 583.66 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 578.30 | 582.64 | 582.93 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 584.10 | 582.07 | 581.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 590.69 | 583.80 | 582.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 606.92 | 609.19 | 602.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 606.92 | 609.19 | 602.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 606.92 | 609.19 | 602.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 604.41 | 609.19 | 602.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 596.27 | 606.61 | 602.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 596.27 | 606.61 | 602.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 598.59 | 605.01 | 601.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 591.52 | 605.01 | 601.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 593.14 | 601.59 | 600.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 593.14 | 601.59 | 600.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 592.88 | 599.84 | 599.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 15:15:00 | 590.80 | 594.09 | 596.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 537.25 | 530.37 | 540.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 537.25 | 530.37 | 540.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 537.20 | 532.14 | 539.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 537.20 | 532.14 | 539.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 539.25 | 533.56 | 539.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:15:00 | 542.75 | 533.56 | 539.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 544.05 | 535.66 | 540.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 544.05 | 535.66 | 540.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 546.75 | 537.88 | 540.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 532.35 | 537.88 | 540.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 550.25 | 537.24 | 538.08 | SL hit (close>static) qty=1.00 sl=547.50 alert=retest2 |

### Cycle 139 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 549.05 | 539.60 | 539.07 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 530.25 | 539.24 | 540.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 525.65 | 536.52 | 539.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 12:15:00 | 521.95 | 519.24 | 524.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-21 13:00:00 | 521.95 | 519.24 | 524.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 523.20 | 520.03 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:00:00 | 523.20 | 520.03 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 517.50 | 519.58 | 522.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 12:15:00 | 515.85 | 519.32 | 522.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 12:45:00 | 515.95 | 518.73 | 521.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:15:00 | 514.85 | 518.73 | 521.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 512.60 | 519.31 | 521.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 522.10 | 519.87 | 521.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 520.50 | 519.87 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 524.35 | 520.77 | 521.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 521.50 | 520.77 | 521.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 532.45 | 523.78 | 522.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 532.45 | 523.78 | 522.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 10:15:00 | 534.75 | 529.41 | 526.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 530.35 | 531.05 | 527.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 530.35 | 531.05 | 527.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 517.05 | 528.25 | 526.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 517.05 | 528.25 | 526.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 524.65 | 527.53 | 526.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 517.90 | 527.53 | 526.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 521.80 | 526.39 | 526.28 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 523.20 | 525.75 | 526.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 512.85 | 521.79 | 524.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 525.55 | 517.74 | 520.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 525.55 | 517.74 | 520.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 525.55 | 517.74 | 520.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 525.55 | 517.74 | 520.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 539.35 | 522.07 | 522.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 539.35 | 522.07 | 522.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 538.50 | 525.35 | 523.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 547.40 | 534.99 | 529.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 546.10 | 547.77 | 539.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 546.10 | 547.77 | 539.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 540.55 | 546.33 | 540.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 539.95 | 546.33 | 540.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 538.30 | 544.72 | 539.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 538.30 | 544.72 | 539.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 537.70 | 543.32 | 539.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 537.70 | 543.32 | 539.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 540.85 | 542.82 | 539.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 541.95 | 542.82 | 539.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 543.85 | 541.71 | 539.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:45:00 | 541.65 | 544.51 | 542.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:15:00 | 543.60 | 543.87 | 542.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 538.25 | 542.75 | 542.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 538.25 | 542.75 | 542.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 520.50 | 538.30 | 540.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 520.50 | 538.30 | 540.10 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 15:15:00 | 548.05 | 537.42 | 537.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 562.50 | 547.11 | 541.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 572.80 | 573.38 | 562.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 13:45:00 | 573.10 | 573.38 | 562.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 566.05 | 571.18 | 564.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 562.80 | 571.18 | 564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 565.65 | 570.07 | 564.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 566.15 | 570.07 | 564.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 561.40 | 568.34 | 564.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 561.40 | 568.34 | 564.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 560.25 | 566.72 | 563.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 560.45 | 566.72 | 563.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 572.10 | 569.53 | 566.22 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 555.85 | 564.27 | 564.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 550.05 | 560.25 | 562.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 560.65 | 556.48 | 559.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 560.65 | 556.48 | 559.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 560.65 | 556.48 | 559.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 562.25 | 556.48 | 559.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 561.25 | 557.44 | 559.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 553.85 | 557.44 | 559.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 548.95 | 541.80 | 547.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 550.05 | 541.80 | 547.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 548.70 | 543.18 | 547.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 548.70 | 543.18 | 547.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 543.50 | 543.24 | 547.17 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 557.05 | 549.05 | 548.88 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 537.50 | 547.10 | 548.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 533.00 | 544.28 | 546.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 542.05 | 538.48 | 542.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 542.05 | 538.48 | 542.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 542.05 | 538.48 | 542.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 543.55 | 538.48 | 542.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 544.00 | 539.58 | 542.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 544.00 | 539.58 | 542.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 544.60 | 540.59 | 542.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 544.60 | 540.59 | 542.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 548.95 | 542.26 | 543.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 551.25 | 542.26 | 543.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 550.90 | 543.99 | 543.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 553.70 | 547.52 | 545.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 576.70 | 580.68 | 573.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 576.70 | 580.68 | 573.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 576.70 | 580.68 | 573.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 573.60 | 580.68 | 573.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 581.30 | 578.93 | 575.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 575.70 | 578.93 | 575.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 579.95 | 579.14 | 576.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 598.20 | 576.67 | 576.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 630.00 | 632.49 | 632.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 630.00 | 632.49 | 632.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 625.75 | 630.02 | 631.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 635.45 | 629.88 | 630.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 635.45 | 629.88 | 630.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 635.45 | 629.88 | 630.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 635.45 | 629.88 | 630.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 633.50 | 630.60 | 631.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:15:00 | 636.00 | 630.60 | 631.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 633.90 | 631.26 | 631.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:30:00 | 637.90 | 631.26 | 631.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 635.65 | 632.14 | 631.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 14:15:00 | 638.05 | 633.32 | 632.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 631.15 | 635.08 | 633.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 631.15 | 635.08 | 633.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 631.15 | 635.08 | 633.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 631.15 | 635.08 | 633.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 633.25 | 634.72 | 633.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 635.95 | 634.72 | 633.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:45:00 | 637.70 | 635.52 | 634.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 627.95 | 634.32 | 633.97 | SL hit (close<static) qty=1.00 sl=629.10 alert=retest2 |

### Cycle 152 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 624.45 | 632.35 | 633.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 621.30 | 630.14 | 632.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 621.85 | 621.66 | 625.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 14:00:00 | 621.85 | 621.66 | 625.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 635.45 | 625.01 | 625.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 635.45 | 625.01 | 625.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 637.90 | 627.59 | 626.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 643.35 | 632.29 | 629.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 666.30 | 666.99 | 658.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 672.20 | 666.99 | 658.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:30:00 | 670.15 | 669.20 | 662.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 12:00:00 | 671.10 | 669.20 | 662.11 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 14:00:00 | 670.30 | 669.65 | 663.56 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 671.25 | 679.41 | 674.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 671.25 | 679.41 | 674.53 | SL hit (close<ema400) qty=1.00 sl=674.53 alert=retest1 |

### Cycle 154 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 663.35 | 675.06 | 675.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 658.10 | 668.45 | 672.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 15:15:00 | 640.25 | 639.44 | 646.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:15:00 | 645.75 | 639.44 | 646.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 652.35 | 642.02 | 646.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 652.35 | 642.02 | 646.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 654.20 | 644.46 | 647.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:15:00 | 656.10 | 644.46 | 647.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 658.70 | 649.91 | 649.55 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 606.40 | 645.01 | 648.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 591.90 | 628.06 | 640.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 619.00 | 618.81 | 630.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 645.25 | 618.81 | 630.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 644.40 | 623.93 | 632.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 644.30 | 623.93 | 632.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 640.40 | 627.22 | 632.89 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 646.00 | 637.53 | 636.47 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 626.00 | 635.23 | 635.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 621.25 | 632.43 | 634.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 12:15:00 | 632.50 | 632.40 | 633.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 12:15:00 | 632.50 | 632.40 | 633.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 632.50 | 632.40 | 633.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 12:30:00 | 630.35 | 632.40 | 633.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 640.35 | 631.16 | 632.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 640.30 | 631.16 | 632.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 641.80 | 633.29 | 633.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 641.80 | 633.29 | 633.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 645.20 | 635.67 | 634.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 665.15 | 643.24 | 638.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 664.55 | 665.61 | 656.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:30:00 | 664.35 | 665.61 | 656.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 692.50 | 700.85 | 696.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 692.50 | 700.85 | 696.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 697.35 | 700.15 | 696.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 690.45 | 700.15 | 696.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 697.70 | 699.37 | 696.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:45:00 | 703.05 | 700.08 | 697.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 702.30 | 700.83 | 697.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:45:00 | 702.45 | 701.26 | 698.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:00:00 | 700.50 | 700.84 | 698.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 697.50 | 700.18 | 698.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 697.50 | 700.18 | 698.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 697.00 | 699.54 | 698.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 696.95 | 699.54 | 698.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 697.50 | 699.13 | 698.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:30:00 | 695.00 | 699.13 | 698.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 683.50 | 695.97 | 697.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 683.50 | 695.97 | 697.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 674.55 | 689.13 | 693.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 618.25 | 618.20 | 632.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:45:00 | 616.90 | 618.20 | 632.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 622.15 | 609.70 | 615.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 622.15 | 609.70 | 615.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 627.15 | 613.19 | 616.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 627.15 | 613.19 | 616.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 627.50 | 619.47 | 618.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 629.65 | 621.51 | 619.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 620.50 | 622.63 | 620.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 620.50 | 622.63 | 620.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 620.50 | 622.63 | 620.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 620.50 | 622.63 | 620.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 620.15 | 622.14 | 620.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 13:45:00 | 625.20 | 622.68 | 621.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 14:15:00 | 624.95 | 622.68 | 621.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 628.75 | 622.52 | 621.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 627.85 | 623.28 | 621.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 630.20 | 631.57 | 627.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 630.20 | 631.57 | 627.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 630.05 | 631.26 | 628.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:45:00 | 628.05 | 631.26 | 628.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 624.75 | 629.65 | 627.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:30:00 | 622.75 | 629.65 | 627.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 616.40 | 627.00 | 626.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 616.40 | 627.00 | 626.79 | SL hit (close<static) qty=1.00 sl=617.80 alert=retest2 |

### Cycle 162 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 609.90 | 623.58 | 625.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 603.35 | 619.53 | 623.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 626.15 | 610.57 | 615.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 626.15 | 610.57 | 615.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 626.15 | 610.57 | 615.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 631.50 | 610.57 | 615.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 634.00 | 615.25 | 616.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 634.00 | 615.25 | 616.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 646.00 | 621.40 | 619.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 654.80 | 639.56 | 633.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 665.45 | 666.50 | 656.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 665.45 | 666.50 | 656.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 673.30 | 672.73 | 667.77 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 659.70 | 665.46 | 665.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 649.65 | 662.30 | 664.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 658.85 | 658.48 | 661.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:30:00 | 655.80 | 658.48 | 661.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 660.05 | 652.57 | 655.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 660.05 | 652.57 | 655.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 658.20 | 653.69 | 655.67 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 659.35 | 657.04 | 656.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 664.90 | 660.35 | 658.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 661.15 | 661.91 | 660.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 662.40 | 662.01 | 660.51 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 656.05 | 659.71 | 659.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 655.95 | 658.35 | 659.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 653.60 | 653.29 | 655.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 644.05 | 651.44 | 654.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 644.05 | 651.44 | 654.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 640.50 | 651.44 | 654.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 641.75 | 647.42 | 651.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:45:00 | 642.30 | 646.14 | 650.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 641.95 | 640.84 | 645.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 644.60 | 641.97 | 645.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 644.60 | 641.97 | 645.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 645.75 | 642.80 | 645.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 651.90 | 646.67 | 646.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 651.90 | 646.67 | 646.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 657.20 | 649.26 | 647.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 646.05 | 649.59 | 648.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 642.50 | 648.17 | 647.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 642.50 | 648.17 | 647.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 640.95 | 646.73 | 647.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 639.10 | 644.45 | 646.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 648.75 | 644.94 | 645.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 652.65 | 646.48 | 646.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 652.65 | 646.48 | 646.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 651.60 | 647.50 | 647.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 680.80 | 655.76 | 651.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 696.90 | 697.17 | 688.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:45:00 | 696.60 | 697.17 | 688.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 694.95 | 697.61 | 690.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 690.25 | 697.61 | 690.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 687.35 | 694.54 | 691.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 687.35 | 694.54 | 691.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 686.50 | 692.93 | 690.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 686.50 | 692.93 | 690.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 684.60 | 691.26 | 690.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 685.65 | 691.26 | 690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 677.60 | 687.43 | 688.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 675.55 | 683.43 | 686.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 665.00 | 664.93 | 673.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 665.00 | 664.93 | 673.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 672.85 | 667.28 | 672.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 678.85 | 667.28 | 672.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 677.30 | 669.29 | 672.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 678.95 | 669.29 | 672.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 676.15 | 670.66 | 673.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 673.05 | 673.13 | 673.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 676.30 | 663.81 | 662.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 676.30 | 663.81 | 662.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 685.85 | 677.95 | 674.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 701.70 | 705.08 | 699.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 701.70 | 705.08 | 699.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 700.30 | 704.12 | 699.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 700.30 | 704.12 | 699.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 699.30 | 703.16 | 699.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 699.30 | 703.16 | 699.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 696.45 | 701.82 | 699.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 697.00 | 701.82 | 699.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 697.25 | 700.90 | 699.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 695.90 | 700.90 | 699.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 686.95 | 696.82 | 697.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 682.90 | 694.04 | 696.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 680.50 | 679.09 | 683.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 677.70 | 678.81 | 682.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 675.80 | 678.81 | 682.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 675.90 | 676.40 | 680.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:00:00 | 675.80 | 676.28 | 680.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 675.20 | 676.82 | 680.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 676.65 | 676.59 | 679.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 676.65 | 676.59 | 679.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 672.60 | 673.00 | 676.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 669.65 | 672.25 | 675.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 668.35 | 671.06 | 674.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 679.05 | 672.95 | 673.97 | SL hit (close>static) qty=1.00 sl=677.40 alert=retest2 |

### Cycle 173 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 682.10 | 675.91 | 675.21 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 673.65 | 676.31 | 676.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 664.95 | 672.76 | 674.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 669.70 | 669.32 | 671.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:15:00 | 668.75 | 669.32 | 671.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 671.00 | 669.36 | 671.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 671.00 | 669.36 | 671.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 668.85 | 669.26 | 670.91 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 679.50 | 672.74 | 671.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 683.15 | 676.86 | 674.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 667.75 | 672.93 | 673.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 660.40 | 665.83 | 668.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 651.70 | 650.13 | 656.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 651.70 | 650.13 | 656.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 656.60 | 651.71 | 655.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 656.60 | 651.71 | 655.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 655.70 | 652.51 | 655.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 650.15 | 652.51 | 655.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 647.85 | 651.58 | 655.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 645.80 | 650.42 | 654.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 645.95 | 649.22 | 653.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 636.35 | 648.33 | 648.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 613.51 | 633.60 | 640.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 613.65 | 633.60 | 640.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 604.53 | 623.66 | 634.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 637.25 | 620.30 | 627.79 | SL hit (close>ema200) qty=0.50 sl=620.30 alert=retest2 |

### Cycle 177 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 639.70 | 631.18 | 630.57 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 626.85 | 632.21 | 632.47 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 636.80 | 633.22 | 632.81 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 623.40 | 631.51 | 632.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 619.40 | 625.51 | 628.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 621.80 | 621.40 | 625.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 621.80 | 621.40 | 625.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 624.65 | 622.44 | 624.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 624.90 | 622.44 | 624.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 625.40 | 623.03 | 624.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 624.10 | 623.03 | 624.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 623.35 | 623.10 | 624.65 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 631.75 | 625.59 | 625.29 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 619.05 | 626.63 | 626.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 616.60 | 623.54 | 624.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 12:15:00 | 616.15 | 614.83 | 618.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 13:00:00 | 616.15 | 614.83 | 618.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 618.05 | 615.48 | 618.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 618.05 | 615.48 | 618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 617.45 | 615.87 | 617.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 617.45 | 615.87 | 617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 616.90 | 616.08 | 617.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 614.80 | 615.98 | 617.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 613.85 | 615.98 | 617.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:45:00 | 614.85 | 613.71 | 615.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 619.50 | 616.21 | 616.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 619.50 | 616.21 | 616.01 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 612.85 | 615.73 | 615.89 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 633.75 | 619.16 | 617.30 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 615.80 | 622.12 | 622.19 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 621.75 | 618.85 | 618.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 623.15 | 619.71 | 618.96 | Break + close above crossover candle high |

### Cycle 188 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 609.00 | 617.84 | 618.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 606.30 | 615.54 | 617.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 585.75 | 581.08 | 590.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 584.50 | 581.08 | 590.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 588.85 | 584.32 | 587.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 588.85 | 584.32 | 587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 588.75 | 585.21 | 587.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 588.75 | 585.21 | 587.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 588.90 | 585.95 | 587.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 589.80 | 585.95 | 587.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 592.60 | 587.53 | 587.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 594.10 | 587.53 | 587.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 591.50 | 588.32 | 588.24 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 583.95 | 588.32 | 588.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 581.30 | 585.95 | 587.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 13:15:00 | 585.10 | 584.45 | 586.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 585.10 | 584.45 | 586.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 586.30 | 584.82 | 586.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 586.30 | 584.82 | 586.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 587.25 | 585.31 | 586.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 589.10 | 585.31 | 586.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 587.35 | 585.72 | 586.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 586.90 | 585.95 | 586.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:15:00 | 586.40 | 586.26 | 586.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:00:00 | 586.55 | 586.36 | 586.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 591.30 | 587.35 | 586.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 591.30 | 587.35 | 586.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 592.40 | 589.00 | 587.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 596.10 | 597.94 | 594.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 596.10 | 597.94 | 594.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 591.00 | 596.34 | 594.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 591.00 | 596.34 | 594.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 592.65 | 595.60 | 593.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:45:00 | 594.30 | 594.75 | 593.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 595.65 | 594.75 | 593.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 594.30 | 594.66 | 593.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 615.50 | 620.42 | 620.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 615.50 | 620.42 | 620.76 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 622.70 | 621.14 | 620.96 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 619.70 | 620.80 | 620.84 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 622.70 | 621.18 | 621.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 626.00 | 622.14 | 621.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 11:15:00 | 632.00 | 632.69 | 629.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 632.00 | 632.69 | 629.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 632.40 | 632.63 | 630.04 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 623.90 | 629.06 | 629.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 618.50 | 624.34 | 626.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 611.50 | 610.64 | 615.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 611.50 | 610.64 | 615.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 615.10 | 610.99 | 613.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 615.10 | 610.99 | 613.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 612.30 | 611.25 | 613.78 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 618.70 | 615.31 | 615.10 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 612.90 | 614.87 | 614.94 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 617.85 | 615.55 | 615.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 627.95 | 618.03 | 616.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 664.90 | 665.64 | 656.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:30:00 | 664.85 | 665.64 | 656.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 665.75 | 666.81 | 663.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 672.00 | 668.26 | 666.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 672.95 | 669.56 | 667.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 671.80 | 670.92 | 669.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 664.50 | 672.17 | 672.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 664.50 | 672.17 | 672.31 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 676.05 | 672.56 | 672.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 682.75 | 676.49 | 674.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 712.80 | 713.20 | 704.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:45:00 | 713.75 | 713.20 | 704.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 711.35 | 718.35 | 714.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 711.35 | 718.35 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 714.85 | 717.65 | 714.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 719.45 | 718.08 | 714.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 09:15:00 | 791.40 | 751.96 | 741.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 814.65 | 818.83 | 819.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 809.50 | 815.56 | 817.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 823.35 | 813.36 | 814.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 825.15 | 815.72 | 815.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 830.35 | 824.17 | 822.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 821.75 | 823.69 | 822.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 822.15 | 823.38 | 822.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 828.90 | 823.38 | 822.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 834.80 | 825.67 | 823.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 838.90 | 825.67 | 823.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 835.20 | 830.47 | 827.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 835.00 | 831.37 | 828.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 851.95 | 854.12 | 854.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 851.95 | 854.12 | 854.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 844.20 | 851.42 | 852.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 828.35 | 828.28 | 833.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 15:15:00 | 831.05 | 828.28 | 833.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 849.40 | 832.95 | 834.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 849.40 | 832.95 | 834.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 848.60 | 836.08 | 836.18 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 850.00 | 838.86 | 837.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 12:15:00 | 853.75 | 841.84 | 838.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 847.60 | 847.96 | 843.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 847.60 | 847.96 | 843.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 841.60 | 846.15 | 843.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 841.60 | 846.15 | 843.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 840.85 | 845.09 | 843.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 840.85 | 845.09 | 843.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 832.35 | 842.54 | 842.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 832.35 | 842.54 | 842.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 834.55 | 840.94 | 841.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 827.75 | 837.09 | 839.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 842.20 | 838.11 | 839.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 841.90 | 838.87 | 839.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 843.00 | 838.87 | 839.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 846.60 | 841.02 | 840.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 850.60 | 844.27 | 842.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 843.35 | 844.34 | 842.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:30:00 | 844.60 | 844.34 | 842.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 841.55 | 843.78 | 842.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 841.35 | 843.78 | 842.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 839.20 | 842.87 | 842.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 839.20 | 842.87 | 842.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 837.30 | 841.75 | 841.91 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 848.80 | 842.40 | 842.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 853.25 | 847.74 | 845.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 855.40 | 855.80 | 851.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 855.40 | 855.80 | 851.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 851.05 | 854.85 | 851.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 850.10 | 854.85 | 851.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 850.70 | 854.02 | 851.42 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 845.80 | 850.39 | 850.57 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 860.40 | 852.39 | 851.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 907.15 | 873.06 | 866.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 959.90 | 966.91 | 951.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 959.90 | 966.91 | 951.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 955.05 | 960.84 | 955.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 955.05 | 960.84 | 955.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 956.40 | 959.95 | 955.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 969.00 | 956.14 | 955.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 993.65 | 1001.69 | 1002.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 993.65 | 1001.69 | 1002.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 988.70 | 995.59 | 998.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 12:15:00 | 994.20 | 993.47 | 996.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 13:00:00 | 994.20 | 993.47 | 996.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 993.00 | 993.38 | 996.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:45:00 | 994.50 | 993.38 | 996.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 993.05 | 993.31 | 996.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 995.80 | 993.31 | 996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 970.05 | 972.27 | 978.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 980.50 | 972.27 | 978.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 978.95 | 973.60 | 978.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:45:00 | 979.75 | 973.60 | 978.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 982.05 | 975.29 | 978.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 984.15 | 975.29 | 978.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 975.35 | 975.30 | 978.52 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 986.30 | 979.71 | 979.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1006.20 | 985.44 | 982.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 995.05 | 997.06 | 990.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 995.05 | 997.06 | 990.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1002.00 | 998.03 | 992.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 1012.40 | 1001.31 | 994.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 984.85 | 998.70 | 998.56 | SL hit (close<static) qty=1.00 sl=991.20 alert=retest2 |

### Cycle 214 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 985.25 | 996.01 | 997.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 982.70 | 993.35 | 996.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1002.15 | 988.94 | 992.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1001.45 | 991.44 | 992.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1002.00 | 991.44 | 992.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 996.80 | 992.51 | 993.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 992.15 | 992.51 | 993.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 1000.60 | 994.97 | 994.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 1000.60 | 994.97 | 994.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1004.70 | 997.61 | 995.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:30:00 | 1007.70 | 1005.00 | 1001.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1002.90 | 1004.58 | 1001.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 991.15 | 1004.58 | 1001.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 999.65 | 1003.59 | 1001.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 989.30 | 1003.59 | 1001.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 996.15 | 1002.10 | 1000.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 1000.15 | 1002.10 | 1000.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 995.70 | 1000.82 | 1000.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 995.70 | 1000.82 | 1000.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 991.65 | 998.99 | 999.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 987.70 | 996.73 | 998.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1000.90 | 997.56 | 998.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1000.05 | 998.06 | 998.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1015.00 | 998.06 | 998.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1004.70 | 999.81 | 999.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1008.55 | 1001.56 | 1000.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 1019.30 | 1019.52 | 1014.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 1019.30 | 1019.52 | 1014.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1018.55 | 1020.27 | 1016.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 1018.15 | 1020.27 | 1016.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1021.00 | 1020.42 | 1016.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 1019.45 | 1020.42 | 1016.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1020.00 | 1020.34 | 1016.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 1019.30 | 1020.34 | 1016.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1019.40 | 1020.15 | 1017.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 1019.40 | 1020.15 | 1017.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1021.00 | 1020.32 | 1017.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1020.00 | 1020.32 | 1017.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 994.70 | 1015.19 | 1015.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 964.50 | 995.93 | 1005.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1013.70 | 991.61 | 991.27 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 986.80 | 995.02 | 995.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 982.80 | 990.51 | 993.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 992.10 | 990.83 | 993.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 994.00 | 991.46 | 993.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 993.30 | 991.46 | 993.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 987.80 | 990.73 | 992.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 983.80 | 990.73 | 992.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 985.80 | 988.01 | 990.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 14:15:00 | 1004.40 | 991.65 | 991.90 | SL hit (close>static) qty=1.00 sl=998.50 alert=retest2 |

### Cycle 221 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 1001.00 | 993.52 | 992.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1019.70 | 998.76 | 995.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1067.80 | 1072.57 | 1060.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:30:00 | 1071.10 | 1072.57 | 1060.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1064.20 | 1069.85 | 1062.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 1064.20 | 1069.85 | 1062.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1069.20 | 1069.72 | 1063.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1070.60 | 1069.72 | 1063.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1074.60 | 1068.01 | 1063.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 1069.90 | 1069.43 | 1064.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1073.10 | 1070.09 | 1068.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1070.40 | 1073.91 | 1072.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 1069.80 | 1073.91 | 1072.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1069.90 | 1073.11 | 1072.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1069.90 | 1073.11 | 1072.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1067.10 | 1071.91 | 1071.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1067.10 | 1071.91 | 1071.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 1069.10 | 1071.35 | 1071.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1069.10 | 1071.35 | 1071.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1065.60 | 1070.20 | 1070.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1060.50 | 1060.37 | 1064.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:30:00 | 1059.70 | 1060.37 | 1064.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1058.40 | 1059.98 | 1064.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1053.30 | 1063.26 | 1063.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1055.60 | 1059.94 | 1062.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 1053.30 | 1059.39 | 1061.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 1055.50 | 1058.86 | 1061.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1061.10 | 1059.15 | 1060.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 1061.50 | 1059.15 | 1060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1065.50 | 1060.42 | 1061.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 1065.50 | 1060.42 | 1061.39 | SL hit (close>static) qty=1.00 sl=1064.80 alert=retest2 |

### Cycle 223 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1075.60 | 1063.46 | 1062.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1081.00 | 1066.97 | 1064.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 1083.10 | 1092.86 | 1084.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1084.80 | 1091.25 | 1084.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1082.80 | 1091.25 | 1084.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1080.10 | 1089.02 | 1084.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 1080.10 | 1089.02 | 1084.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1078.30 | 1086.87 | 1083.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 1078.30 | 1086.87 | 1083.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1080.20 | 1085.08 | 1083.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1080.20 | 1085.08 | 1083.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1079.50 | 1083.96 | 1083.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1061.50 | 1083.96 | 1083.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1060.00 | 1079.17 | 1081.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1000.00 | 1046.61 | 1061.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1035.10 | 1021.92 | 1038.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 1035.10 | 1021.92 | 1038.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1037.60 | 1025.06 | 1038.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 1037.60 | 1025.06 | 1038.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1031.50 | 1026.35 | 1037.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1028.10 | 1027.53 | 1036.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1039.30 | 1029.88 | 1036.43 | SL hit (close>static) qty=1.00 sl=1037.60 alert=retest2 |

### Cycle 225 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1046.60 | 1015.67 | 1014.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 1055.70 | 1023.68 | 1017.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1040.00 | 1047.30 | 1036.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1040.00 | 1047.30 | 1036.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1033.90 | 1044.62 | 1036.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1033.70 | 1044.62 | 1036.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1029.80 | 1041.66 | 1035.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1029.80 | 1041.66 | 1035.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1038.00 | 1040.93 | 1035.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1014.70 | 1040.93 | 1035.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1016.00 | 1035.94 | 1033.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1011.20 | 1035.94 | 1033.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1035.90 | 1034.96 | 1033.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1035.90 | 1034.96 | 1033.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1030.70 | 1034.78 | 1033.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 1036.50 | 1034.78 | 1033.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1029.10 | 1033.65 | 1033.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 1019.10 | 1033.65 | 1033.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1009.70 | 1028.86 | 1031.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1001.40 | 1023.37 | 1028.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1007.50 | 995.06 | 1003.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1005.90 | 997.23 | 1003.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1000.00 | 997.23 | 1003.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 1000.80 | 999.93 | 1003.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1000.00 | 999.93 | 1003.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1013.10 | 1003.84 | 1004.61 | SL hit (close>static) qty=1.00 sl=1012.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1012.30 | 1005.54 | 1005.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1020.40 | 1010.82 | 1007.97 | Break + close above crossover candle high |

### Cycle 228 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 980.80 | 1007.37 | 1007.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 967.20 | 999.33 | 1003.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 898.80 | 894.90 | 921.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:30:00 | 904.10 | 894.90 | 921.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 916.00 | 903.32 | 918.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 928.40 | 903.32 | 918.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 944.70 | 913.40 | 918.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 941.20 | 913.40 | 918.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 957.50 | 922.22 | 922.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 957.50 | 922.22 | 922.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 954.90 | 928.76 | 925.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 962.20 | 939.68 | 931.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 913.10 | 939.16 | 933.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 915.90 | 934.51 | 931.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 910.80 | 934.51 | 931.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 907.50 | 929.11 | 929.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 903.50 | 918.82 | 924.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 908.25 | 888.52 | 900.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 904.30 | 899.36 | 902.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 905.65 | 901.19 | 903.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 906.20 | 895.91 | 895.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 906.20 | 895.91 | 895.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 916.50 | 900.03 | 897.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 998.70 | 1004.22 | 979.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 998.70 | 1004.22 | 979.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 995.55 | 1014.68 | 1002.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 1010.00 | 1011.30 | 1003.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1024.25 | 1039.16 | 1039.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1024.25 | 1039.16 | 1039.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 1019.20 | 1035.17 | 1037.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1021.10 | 1009.20 | 1017.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1016.00 | 1010.56 | 1017.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 978.20 | 1010.56 | 1017.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 929.29 | 954.81 | 966.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 960.80 | 944.19 | 953.09 | SL hit (close>ema200) qty=0.50 sl=944.19 alert=retest2 |

### Cycle 233 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 962.10 | 956.72 | 956.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 965.00 | 960.00 | 957.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1006.70 | 1007.77 | 994.14 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 12:00:00 | 268.67 | 2023-05-18 15:15:00 | 270.25 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-05-18 09:15:00 | 268.20 | 2023-05-18 15:15:00 | 270.25 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-05-26 13:30:00 | 275.50 | 2023-06-05 14:15:00 | 280.17 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2023-05-26 14:00:00 | 275.60 | 2023-06-05 14:15:00 | 280.17 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2023-06-12 12:30:00 | 287.80 | 2023-06-14 12:15:00 | 282.87 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-06-12 14:30:00 | 287.71 | 2023-06-14 12:15:00 | 282.87 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-06-13 11:15:00 | 287.60 | 2023-06-14 12:15:00 | 282.87 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-06-26 14:00:00 | 335.89 | 2023-07-07 09:15:00 | 349.87 | STOP_HIT | 1.00 | 4.16% |
| BUY | retest2 | 2023-06-27 13:45:00 | 337.24 | 2023-07-07 09:15:00 | 349.87 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2023-07-25 09:15:00 | 361.86 | 2023-07-28 09:15:00 | 361.37 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-07-25 13:15:00 | 361.16 | 2023-07-28 09:15:00 | 361.37 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2023-08-10 09:30:00 | 372.29 | 2023-08-11 09:15:00 | 367.69 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-08-10 10:30:00 | 372.27 | 2023-08-11 09:15:00 | 367.69 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-08-16 14:45:00 | 359.38 | 2023-08-17 09:15:00 | 369.77 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2023-09-12 11:30:00 | 387.68 | 2023-09-12 12:15:00 | 380.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-09-14 12:15:00 | 382.63 | 2023-09-15 14:15:00 | 384.19 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-09-14 13:30:00 | 382.80 | 2023-09-15 14:15:00 | 384.19 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-09-14 15:15:00 | 382.41 | 2023-09-15 14:15:00 | 384.19 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-09-15 12:15:00 | 382.85 | 2023-09-15 14:15:00 | 384.19 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2023-10-16 13:30:00 | 374.70 | 2023-10-17 14:15:00 | 377.64 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-10-17 13:30:00 | 374.82 | 2023-10-17 14:15:00 | 377.64 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-11-13 09:30:00 | 388.74 | 2023-11-13 13:15:00 | 393.55 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-11-21 10:30:00 | 397.20 | 2023-11-21 12:15:00 | 403.20 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-11-23 11:00:00 | 394.87 | 2023-11-29 12:15:00 | 394.91 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2023-11-23 13:45:00 | 395.18 | 2023-11-29 12:15:00 | 394.91 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2023-11-23 15:00:00 | 394.97 | 2023-11-29 12:15:00 | 394.91 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2023-12-06 15:15:00 | 412.40 | 2023-12-07 11:15:00 | 408.60 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-12-07 09:45:00 | 412.25 | 2023-12-07 11:15:00 | 408.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-12-22 12:30:00 | 409.04 | 2023-12-22 15:15:00 | 411.12 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-01-20 09:30:00 | 463.00 | 2024-01-20 15:15:00 | 458.77 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest1 | 2024-01-20 11:30:00 | 461.91 | 2024-01-20 15:15:00 | 458.77 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-02-01 12:00:00 | 482.77 | 2024-02-01 15:15:00 | 478.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-02-01 13:00:00 | 483.47 | 2024-02-01 15:15:00 | 478.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-02-01 13:45:00 | 482.69 | 2024-02-01 15:15:00 | 478.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-02-14 09:15:00 | 462.51 | 2024-02-14 09:15:00 | 469.63 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-02-27 12:30:00 | 492.41 | 2024-02-27 13:15:00 | 462.90 | STOP_HIT | 1.00 | -5.99% |
| BUY | retest2 | 2024-03-12 13:15:00 | 491.07 | 2024-03-12 14:15:00 | 484.46 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2024-03-15 09:30:00 | 459.79 | 2024-03-18 09:15:00 | 461.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-03-22 09:15:00 | 465.42 | 2024-04-03 09:15:00 | 511.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-22 14:15:00 | 464.96 | 2024-04-03 09:15:00 | 511.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-26 10:15:00 | 463.94 | 2024-04-03 09:15:00 | 510.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-26 11:00:00 | 463.99 | 2024-04-03 09:15:00 | 510.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-01 09:15:00 | 481.20 | 2024-04-10 13:15:00 | 506.34 | STOP_HIT | 1.00 | 5.22% |
| BUY | retest2 | 2024-04-01 11:00:00 | 481.64 | 2024-04-10 13:15:00 | 506.34 | STOP_HIT | 1.00 | 5.13% |
| SELL | retest2 | 2024-04-18 14:45:00 | 477.02 | 2024-04-22 14:15:00 | 485.72 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-04-24 11:45:00 | 490.65 | 2024-04-26 12:15:00 | 472.45 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-04-25 12:45:00 | 492.93 | 2024-04-26 12:15:00 | 472.45 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2024-05-15 09:15:00 | 464.49 | 2024-05-17 11:15:00 | 469.17 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-05-15 10:00:00 | 462.81 | 2024-05-17 11:15:00 | 469.17 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-15 13:15:00 | 464.59 | 2024-05-17 11:15:00 | 469.17 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-05-15 13:45:00 | 464.57 | 2024-05-17 11:15:00 | 469.17 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-05-24 12:15:00 | 480.40 | 2024-05-29 11:15:00 | 476.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-27 10:00:00 | 479.37 | 2024-05-29 11:15:00 | 476.40 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-05-28 09:30:00 | 480.00 | 2024-05-29 11:15:00 | 476.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-28 10:30:00 | 478.85 | 2024-05-29 11:15:00 | 476.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-05-31 14:15:00 | 466.26 | 2024-05-31 15:15:00 | 477.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-06-20 11:45:00 | 564.96 | 2024-06-28 13:15:00 | 586.92 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2024-06-21 09:15:00 | 563.92 | 2024-06-28 13:15:00 | 586.92 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2024-06-21 10:15:00 | 565.11 | 2024-06-28 13:15:00 | 586.92 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2024-07-02 11:15:00 | 570.89 | 2024-07-10 10:15:00 | 542.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-03 13:45:00 | 571.75 | 2024-07-10 10:15:00 | 543.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-03 14:15:00 | 571.87 | 2024-07-10 10:15:00 | 543.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-02 11:15:00 | 570.89 | 2024-07-11 15:15:00 | 549.03 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2024-07-03 13:45:00 | 571.75 | 2024-07-11 15:15:00 | 549.03 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2024-07-03 14:15:00 | 571.87 | 2024-07-11 15:15:00 | 549.03 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2024-07-04 09:15:00 | 569.24 | 2024-07-12 10:15:00 | 564.52 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-07-08 09:15:00 | 564.95 | 2024-07-12 10:15:00 | 564.52 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-07-18 13:00:00 | 569.56 | 2024-07-19 13:15:00 | 565.13 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-19 10:15:00 | 569.00 | 2024-07-19 13:15:00 | 565.13 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-07-22 15:00:00 | 565.30 | 2024-07-22 15:15:00 | 566.12 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-09-03 14:45:00 | 655.54 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-09-04 10:00:00 | 654.71 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-04 10:45:00 | 653.35 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-09-05 09:15:00 | 657.54 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-05 15:15:00 | 649.99 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-09-06 09:30:00 | 649.89 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-06 10:45:00 | 650.32 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-09-06 11:15:00 | 649.61 | 2024-09-06 15:15:00 | 647.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-11 09:30:00 | 654.59 | 2024-09-19 09:15:00 | 720.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-11 11:45:00 | 654.47 | 2024-09-19 09:15:00 | 719.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-11 15:00:00 | 655.85 | 2024-09-19 09:15:00 | 721.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-01 09:30:00 | 722.79 | 2024-10-01 12:15:00 | 716.71 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-10-21 09:30:00 | 665.11 | 2024-10-22 10:15:00 | 668.50 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-10-22 10:15:00 | 664.13 | 2024-10-22 10:15:00 | 668.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-10-25 09:15:00 | 641.96 | 2024-10-25 13:15:00 | 609.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 641.96 | 2024-10-28 09:15:00 | 649.89 | STOP_HIT | 0.50 | -1.24% |
| SELL | retest2 | 2024-10-28 09:30:00 | 646.67 | 2024-10-28 10:15:00 | 658.96 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-04 15:15:00 | 628.40 | 2024-11-06 10:15:00 | 633.19 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-11-19 10:45:00 | 566.77 | 2024-11-22 14:15:00 | 570.83 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-19 12:00:00 | 566.78 | 2024-11-22 14:15:00 | 570.83 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-11-19 15:00:00 | 565.31 | 2024-11-22 14:15:00 | 570.83 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-11-29 12:15:00 | 604.60 | 2024-12-09 10:15:00 | 621.14 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2024-11-29 13:00:00 | 606.08 | 2024-12-09 10:15:00 | 621.14 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2024-12-02 09:15:00 | 611.16 | 2024-12-09 10:15:00 | 621.14 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2024-12-23 11:15:00 | 583.56 | 2024-12-26 14:15:00 | 586.56 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-26 13:45:00 | 582.56 | 2024-12-26 14:15:00 | 586.56 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-01-15 09:15:00 | 532.35 | 2025-01-16 09:15:00 | 550.25 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-01-22 12:15:00 | 515.85 | 2025-01-23 12:15:00 | 532.45 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-01-22 12:45:00 | 515.95 | 2025-01-23 12:15:00 | 532.45 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-01-22 13:15:00 | 514.85 | 2025-01-23 12:15:00 | 532.45 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-01-23 09:15:00 | 512.60 | 2025-01-23 12:15:00 | 532.45 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-01-23 11:15:00 | 521.50 | 2025-01-23 12:15:00 | 532.45 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-01-31 09:15:00 | 541.95 | 2025-02-01 12:15:00 | 520.50 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-01-31 10:15:00 | 543.85 | 2025-02-01 12:15:00 | 520.50 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2025-02-01 09:45:00 | 541.65 | 2025-02-01 12:15:00 | 520.50 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-02-01 11:15:00 | 543.60 | 2025-02-01 12:15:00 | 520.50 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-02-27 09:15:00 | 598.20 | 2025-03-10 10:15:00 | 630.00 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2025-03-12 13:15:00 | 635.95 | 2025-03-13 09:15:00 | 627.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-03-12 13:45:00 | 637.70 | 2025-03-13 09:15:00 | 627.95 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2025-03-21 09:15:00 | 672.20 | 2025-03-25 09:15:00 | 671.25 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-03-21 11:30:00 | 670.15 | 2025-03-25 09:15:00 | 671.25 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest1 | 2025-03-21 12:00:00 | 671.10 | 2025-03-25 09:15:00 | 671.25 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2025-03-21 14:00:00 | 670.30 | 2025-03-25 09:15:00 | 671.25 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-03-25 13:30:00 | 677.10 | 2025-03-28 09:15:00 | 663.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-03-26 10:00:00 | 676.60 | 2025-03-28 09:15:00 | 663.35 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-03-27 09:15:00 | 676.35 | 2025-03-28 09:15:00 | 663.35 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-03-27 13:45:00 | 675.10 | 2025-03-28 09:15:00 | 663.35 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-04-23 13:45:00 | 703.05 | 2025-04-25 09:15:00 | 683.50 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-04-23 14:45:00 | 702.30 | 2025-04-25 09:15:00 | 683.50 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-04-24 09:45:00 | 702.45 | 2025-04-25 09:15:00 | 683.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-04-24 12:00:00 | 700.50 | 2025-04-25 09:15:00 | 683.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-05-06 13:45:00 | 625.20 | 2025-05-08 14:15:00 | 616.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-06 14:15:00 | 624.95 | 2025-05-08 14:15:00 | 616.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-05-07 09:15:00 | 628.75 | 2025-05-08 14:15:00 | 616.40 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-07 11:00:00 | 627.85 | 2025-05-08 14:15:00 | 616.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-05-30 10:15:00 | 640.50 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-30 12:30:00 | 641.75 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-30 13:45:00 | 642.30 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-02 12:45:00 | 641.95 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-17 09:45:00 | 673.05 | 2025-06-24 09:15:00 | 676.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-04 11:15:00 | 675.80 | 2025-07-09 10:15:00 | 679.05 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-04 14:00:00 | 675.90 | 2025-07-09 10:15:00 | 679.05 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-04 15:00:00 | 675.80 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-07 09:15:00 | 675.20 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-08 11:30:00 | 669.65 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-08 14:00:00 | 668.35 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-07-22 11:00:00 | 645.80 | 2025-07-25 09:15:00 | 613.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 645.95 | 2025-07-25 09:15:00 | 613.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:15:00 | 636.35 | 2025-07-25 11:15:00 | 604.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 645.80 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-07-22 11:30:00 | 645.95 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-07-24 10:15:00 | 636.35 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2025-08-12 09:30:00 | 614.80 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-12 10:15:00 | 613.85 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-08-13 09:45:00 | 614.85 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-04 11:00:00 | 586.90 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-09-04 13:15:00 | 586.40 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-05 10:00:00 | 586.55 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-09 12:45:00 | 594.30 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2025-09-09 13:15:00 | 595.65 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.33% |
| BUY | retest2 | 2025-09-09 14:00:00 | 594.30 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2025-10-13 12:45:00 | 672.00 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-13 15:00:00 | 672.95 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-14 13:15:00 | 671.80 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-28 11:30:00 | 719.45 | 2025-11-03 09:15:00 | 791.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:15:00 | 838.90 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-11-25 11:15:00 | 835.20 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-11-25 11:45:00 | 835.00 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2025-12-30 09:15:00 | 969.00 | 2026-01-06 12:15:00 | 993.65 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2026-01-19 12:15:00 | 1012.40 | 2026-01-20 14:15:00 | 984.85 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-22 11:15:00 | 992.15 | 2026-01-22 12:15:00 | 1000.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-06 10:15:00 | 983.80 | 2026-02-06 14:15:00 | 1004.40 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-02-06 12:15:00 | 985.80 | 2026-02-06 14:15:00 | 1004.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-13 14:15:00 | 1070.60 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1074.60 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-16 10:30:00 | 1069.90 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-02-17 14:30:00 | 1073.10 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1053.30 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-02-24 11:00:00 | 1055.60 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1053.30 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-02-24 13:15:00 | 1055.50 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-25 09:15:00 | 1059.30 | 2026-02-25 09:15:00 | 1075.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1028.10 | 2026-03-05 14:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-06 13:45:00 | 1026.50 | 2026-03-09 09:15:00 | 975.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 1026.50 | 2026-03-09 15:15:00 | 989.90 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1000.00 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-17 13:30:00 | 1000.80 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-03-17 14:15:00 | 1000.00 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-01 10:15:00 | 908.25 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2026-04-01 13:15:00 | 904.30 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-04-01 14:15:00 | 905.65 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-04-13 12:00:00 | 1010.00 | 2026-04-23 09:15:00 | 1024.25 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2026-04-27 09:15:00 | 978.20 | 2026-04-30 09:15:00 | 929.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 09:15:00 | 978.20 | 2026-05-04 09:15:00 | 960.80 | STOP_HIT | 0.50 | 1.78% |
