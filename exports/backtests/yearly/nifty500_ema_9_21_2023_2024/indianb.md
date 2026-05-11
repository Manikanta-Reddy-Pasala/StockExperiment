# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 862.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 211 |
| ALERT1 | 141 |
| ALERT2 | 140 |
| ALERT2_SKIP | 84 |
| ALERT3 | 392 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 167 |
| PARTIAL | 29 |
| TARGET_HIT | 8 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 198 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 98 / 100
- **Target hits / Stop hits / Partials:** 8 / 161 / 29
- **Avg / median % per leg:** 1.17% / -0.22%
- **Sum % (uncompounded):** 230.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 31 | 36.9% | 8 | 75 | 1 | 0.77% | 64.4% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 5.22% | 15.7% |
| BUY @ 3rd Alert (retest2) | 81 | 28 | 34.6% | 7 | 74 | 0 | 0.60% | 48.7% |
| SELL (all) | 114 | 67 | 58.8% | 0 | 86 | 28 | 1.46% | 166.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 114 | 67 | 58.8% | 0 | 86 | 28 | 1.46% | 166.3% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 5.22% | 15.7% |
| retest2 (combined) | 195 | 95 | 48.7% | 7 | 160 | 28 | 1.10% | 215.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 11:15:00 | 295.05 | 292.23 | 292.16 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 288.80 | 292.58 | 292.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 288.00 | 291.67 | 292.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 12:15:00 | 289.35 | 288.90 | 290.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 12:15:00 | 289.35 | 288.90 | 290.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 12:15:00 | 289.35 | 288.90 | 290.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 12:45:00 | 289.50 | 288.90 | 290.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 287.20 | 288.56 | 289.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 09:15:00 | 285.70 | 288.34 | 289.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 09:45:00 | 287.00 | 285.85 | 287.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 288.50 | 287.66 | 287.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 288.50 | 287.66 | 287.60 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 287.00 | 287.57 | 287.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 286.25 | 287.30 | 287.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 272.90 | 270.61 | 272.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 272.90 | 270.61 | 272.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 272.90 | 270.61 | 272.23 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 14:15:00 | 275.00 | 273.20 | 273.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 278.20 | 274.60 | 273.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 12:15:00 | 280.00 | 280.85 | 278.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 14:15:00 | 278.90 | 280.32 | 278.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 278.90 | 280.32 | 278.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:45:00 | 279.00 | 280.32 | 278.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 279.00 | 280.06 | 278.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:30:00 | 279.70 | 279.83 | 278.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 10:15:00 | 276.55 | 279.17 | 278.59 | SL hit (close<static) qty=1.00 sl=278.10 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 10:15:00 | 277.95 | 282.32 | 282.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 11:15:00 | 277.10 | 281.27 | 282.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 280.25 | 279.32 | 280.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 280.25 | 279.32 | 280.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 280.25 | 279.32 | 280.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:45:00 | 280.35 | 279.32 | 280.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 279.50 | 279.35 | 280.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 11:30:00 | 280.50 | 279.35 | 280.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 281.80 | 279.80 | 280.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 15:00:00 | 281.80 | 279.80 | 280.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 281.80 | 280.20 | 280.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 09:15:00 | 282.30 | 280.20 | 280.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 281.75 | 280.85 | 280.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 10:15:00 | 282.75 | 281.78 | 281.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 12:15:00 | 281.50 | 281.83 | 281.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 12:15:00 | 281.50 | 281.83 | 281.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 281.50 | 281.83 | 281.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:30:00 | 281.70 | 281.83 | 281.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 281.95 | 281.85 | 281.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:30:00 | 282.25 | 281.85 | 281.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 280.40 | 281.56 | 281.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:30:00 | 279.95 | 281.56 | 281.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 281.05 | 281.46 | 281.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:15:00 | 278.95 | 281.46 | 281.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 09:15:00 | 277.75 | 280.72 | 281.04 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 285.30 | 281.59 | 281.13 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 12:15:00 | 281.75 | 282.22 | 282.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 280.05 | 281.67 | 281.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 10:15:00 | 279.75 | 279.44 | 280.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 10:30:00 | 279.55 | 279.44 | 280.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 280.00 | 278.95 | 279.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:30:00 | 279.55 | 278.95 | 279.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 280.25 | 279.21 | 279.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 12:30:00 | 280.30 | 279.21 | 279.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 280.55 | 279.48 | 279.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 280.55 | 279.48 | 279.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 282.25 | 280.03 | 279.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 15:15:00 | 282.45 | 280.52 | 280.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 10:15:00 | 280.65 | 280.82 | 280.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 11:00:00 | 280.65 | 280.82 | 280.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 281.40 | 280.94 | 280.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:30:00 | 280.85 | 280.94 | 280.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 279.95 | 280.82 | 280.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:00:00 | 279.95 | 280.82 | 280.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 279.00 | 280.46 | 280.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:45:00 | 279.00 | 280.46 | 280.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 277.00 | 279.77 | 280.08 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 283.65 | 280.21 | 280.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 12:15:00 | 285.40 | 282.18 | 281.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 15:15:00 | 300.00 | 300.41 | 296.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-05 09:45:00 | 300.40 | 300.77 | 296.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 305.25 | 304.62 | 302.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 12:15:00 | 306.10 | 304.67 | 302.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-24 15:15:00 | 336.71 | 333.24 | 330.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 09:15:00 | 329.25 | 330.67 | 330.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 10:15:00 | 326.35 | 329.80 | 330.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 15:15:00 | 329.50 | 328.88 | 329.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:15:00 | 331.60 | 328.88 | 329.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 331.25 | 329.36 | 329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 333.55 | 329.36 | 329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 329.35 | 329.56 | 329.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:00:00 | 329.35 | 329.56 | 329.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 328.75 | 329.40 | 329.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 15:15:00 | 326.70 | 329.40 | 329.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 09:15:00 | 340.40 | 331.17 | 330.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 340.40 | 331.17 | 330.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 345.00 | 338.24 | 334.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 15:15:00 | 346.05 | 346.56 | 343.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 09:15:00 | 341.15 | 346.56 | 343.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 343.10 | 345.87 | 343.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 341.85 | 345.87 | 343.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 343.50 | 345.40 | 343.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 343.50 | 345.40 | 343.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 343.85 | 345.09 | 343.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 343.45 | 345.09 | 343.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 340.50 | 344.17 | 343.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 340.50 | 344.17 | 343.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 340.05 | 343.35 | 342.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:30:00 | 340.00 | 343.35 | 342.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 342.75 | 343.28 | 342.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:15:00 | 341.80 | 343.28 | 342.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 336.90 | 342.00 | 342.41 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 344.65 | 342.58 | 342.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 346.50 | 343.36 | 342.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 13:15:00 | 343.30 | 343.39 | 342.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 13:15:00 | 343.30 | 343.39 | 342.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 13:15:00 | 343.30 | 343.39 | 342.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 345.95 | 343.35 | 342.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:45:00 | 344.35 | 343.45 | 343.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 10:30:00 | 346.30 | 343.98 | 343.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-08 13:15:00 | 380.55 | 355.66 | 349.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 13:15:00 | 404.55 | 407.70 | 407.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 404.00 | 406.96 | 407.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 401.40 | 397.42 | 400.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 401.40 | 397.42 | 400.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 401.40 | 397.42 | 400.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 10:15:00 | 394.15 | 398.86 | 400.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 09:15:00 | 393.00 | 384.44 | 383.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 393.00 | 384.44 | 383.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 394.00 | 390.94 | 389.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 13:15:00 | 391.70 | 392.02 | 390.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 14:15:00 | 389.05 | 391.43 | 390.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 389.05 | 391.43 | 390.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:45:00 | 388.30 | 391.43 | 390.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 388.10 | 390.76 | 390.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:30:00 | 392.05 | 390.89 | 390.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 11:00:00 | 392.00 | 391.11 | 390.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 12:15:00 | 391.00 | 390.97 | 390.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 383.10 | 392.23 | 391.68 | SL hit (close<static) qty=1.00 sl=387.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 384.65 | 390.15 | 390.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 383.15 | 387.73 | 389.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 387.60 | 385.76 | 387.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 387.60 | 385.76 | 387.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 387.60 | 385.76 | 387.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:00:00 | 387.60 | 385.76 | 387.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 398.60 | 388.33 | 388.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:00:00 | 398.60 | 388.33 | 388.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 397.50 | 390.16 | 389.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 401.35 | 393.42 | 391.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 396.50 | 401.45 | 398.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 396.50 | 401.45 | 398.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 396.50 | 401.45 | 398.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 396.50 | 401.45 | 398.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 396.00 | 400.36 | 398.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 09:15:00 | 405.50 | 400.36 | 398.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-27 10:15:00 | 446.05 | 431.54 | 427.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 410.80 | 427.04 | 428.47 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 431.85 | 426.89 | 426.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 14:15:00 | 439.80 | 431.66 | 429.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 12:15:00 | 436.25 | 436.79 | 433.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 12:15:00 | 436.25 | 436.79 | 433.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 436.25 | 436.79 | 433.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:30:00 | 436.85 | 436.79 | 433.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 433.45 | 436.04 | 433.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 15:00:00 | 433.45 | 436.04 | 433.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 433.95 | 435.62 | 433.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 439.35 | 435.62 | 433.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 10:15:00 | 430.45 | 433.96 | 433.05 | SL hit (close<static) qty=1.00 sl=431.60 alert=retest2 |

### Cycle 24 — SELL (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 12:15:00 | 427.65 | 432.13 | 432.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 425.60 | 429.16 | 430.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 14:15:00 | 410.45 | 409.92 | 415.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 14:45:00 | 410.50 | 409.92 | 415.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 419.15 | 411.77 | 415.00 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 14:15:00 | 424.70 | 417.70 | 417.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 425.50 | 420.10 | 418.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 11:15:00 | 424.50 | 424.52 | 422.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:30:00 | 424.00 | 424.52 | 422.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 423.00 | 424.21 | 422.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:30:00 | 423.00 | 424.21 | 422.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 420.00 | 423.37 | 422.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:45:00 | 420.00 | 423.37 | 422.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 418.00 | 422.30 | 421.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 418.00 | 422.30 | 421.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 428.10 | 427.31 | 425.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 10:15:00 | 434.00 | 428.15 | 426.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 11:30:00 | 432.00 | 428.50 | 426.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 415.65 | 426.09 | 426.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 415.65 | 426.09 | 426.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 15:15:00 | 412.00 | 423.28 | 424.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 425.10 | 422.03 | 423.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 425.10 | 422.03 | 423.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 425.10 | 422.03 | 423.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:00:00 | 425.10 | 422.03 | 423.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 423.00 | 422.22 | 423.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 421.35 | 421.96 | 422.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 420.00 | 421.98 | 422.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 400.28 | 409.42 | 413.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 399.00 | 409.42 | 413.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 411.80 | 404.96 | 408.40 | SL hit (close>ema200) qty=0.50 sl=404.96 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 413.90 | 410.76 | 410.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 416.00 | 411.81 | 410.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 409.60 | 411.37 | 410.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 09:15:00 | 409.60 | 411.37 | 410.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 409.60 | 411.37 | 410.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 09:15:00 | 418.00 | 410.74 | 410.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 10:00:00 | 417.65 | 412.12 | 411.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 14:15:00 | 420.80 | 426.72 | 427.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 420.80 | 426.72 | 427.22 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 427.90 | 426.21 | 426.09 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 14:15:00 | 424.30 | 425.87 | 425.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 15:15:00 | 423.10 | 425.32 | 425.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 13:15:00 | 421.95 | 420.93 | 422.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 13:15:00 | 421.95 | 420.93 | 422.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 421.95 | 420.93 | 422.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:45:00 | 421.50 | 420.93 | 422.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 416.90 | 420.12 | 421.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:30:00 | 421.15 | 420.12 | 421.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 419.20 | 419.52 | 421.23 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 12:15:00 | 438.00 | 424.91 | 423.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 13:15:00 | 440.85 | 428.09 | 424.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 15:15:00 | 445.00 | 445.48 | 438.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 441.25 | 444.64 | 438.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 441.25 | 444.64 | 438.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 441.25 | 444.64 | 438.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 437.95 | 443.29 | 440.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:00:00 | 437.95 | 443.29 | 440.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 438.00 | 442.23 | 440.69 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 12:15:00 | 433.05 | 439.37 | 439.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 13:15:00 | 431.70 | 437.83 | 438.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 15:15:00 | 414.00 | 413.27 | 416.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 09:15:00 | 414.35 | 413.27 | 416.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 415.10 | 413.63 | 416.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 415.10 | 413.63 | 416.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 411.05 | 406.49 | 409.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:00:00 | 411.05 | 406.49 | 409.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 408.65 | 406.92 | 409.22 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 418.65 | 410.61 | 410.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 15:15:00 | 418.90 | 414.51 | 412.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 11:15:00 | 413.65 | 415.80 | 413.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 11:15:00 | 413.65 | 415.80 | 413.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 413.65 | 415.80 | 413.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:00:00 | 413.65 | 415.80 | 413.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 407.85 | 414.21 | 413.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:45:00 | 406.75 | 414.21 | 413.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 408.00 | 412.97 | 412.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:45:00 | 407.25 | 412.97 | 412.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 14:15:00 | 397.35 | 409.84 | 411.34 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 13:15:00 | 409.80 | 407.24 | 407.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 412.05 | 408.87 | 407.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 409.05 | 409.61 | 408.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 409.05 | 409.61 | 408.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 409.05 | 409.61 | 408.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 408.50 | 409.61 | 408.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 411.30 | 409.94 | 408.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:15:00 | 413.45 | 409.89 | 409.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 10:00:00 | 415.05 | 410.92 | 409.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-15 14:15:00 | 454.80 | 445.34 | 441.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 10:15:00 | 440.50 | 444.51 | 444.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 437.90 | 443.12 | 444.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 416.25 | 415.57 | 424.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 10:15:00 | 415.00 | 415.57 | 424.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 417.00 | 413.01 | 416.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 417.00 | 413.01 | 416.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 415.65 | 413.54 | 416.06 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 11:15:00 | 419.80 | 416.88 | 416.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 423.80 | 418.64 | 417.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 418.00 | 418.89 | 417.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 418.00 | 418.89 | 417.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 418.00 | 418.89 | 417.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 418.00 | 418.89 | 417.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 418.70 | 418.85 | 417.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:15:00 | 417.90 | 418.85 | 417.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 418.05 | 418.69 | 417.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:45:00 | 418.05 | 418.69 | 417.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 418.00 | 418.55 | 417.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 12:30:00 | 418.15 | 418.55 | 417.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 418.55 | 418.55 | 418.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 14:15:00 | 418.75 | 418.55 | 418.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 417.55 | 421.04 | 420.29 | SL hit (close<static) qty=1.00 sl=417.90 alert=retest2 |

### Cycle 38 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 411.65 | 419.16 | 419.50 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 13:15:00 | 420.80 | 419.08 | 418.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 425.00 | 420.70 | 419.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 429.85 | 432.82 | 429.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 429.85 | 432.82 | 429.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 429.85 | 432.82 | 429.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:45:00 | 429.65 | 432.82 | 429.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 427.95 | 431.84 | 429.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 427.95 | 431.84 | 429.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 427.70 | 431.01 | 429.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 427.45 | 431.01 | 429.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 428.70 | 428.80 | 428.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:30:00 | 429.40 | 428.80 | 428.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 11:15:00 | 424.20 | 427.88 | 428.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 420.20 | 425.60 | 427.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 13:15:00 | 423.30 | 422.71 | 424.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 14:00:00 | 423.30 | 422.71 | 424.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 423.35 | 422.85 | 424.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:45:00 | 423.00 | 423.00 | 424.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 427.85 | 423.97 | 424.53 | SL hit (close>static) qty=1.00 sl=427.80 alert=retest2 |

### Cycle 41 — BUY (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 15:15:00 | 426.00 | 425.00 | 424.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 430.00 | 426.00 | 425.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 12:15:00 | 429.50 | 432.05 | 430.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 12:15:00 | 429.50 | 432.05 | 430.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 429.50 | 432.05 | 430.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:00:00 | 429.50 | 432.05 | 430.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 433.00 | 432.24 | 430.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 15:00:00 | 435.05 | 432.80 | 430.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 09:15:00 | 438.95 | 445.77 | 445.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 438.95 | 445.77 | 445.83 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 470.15 | 448.27 | 446.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 14:15:00 | 480.50 | 469.49 | 461.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 15:15:00 | 495.90 | 497.03 | 487.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 12:00:00 | 507.65 | 499.30 | 491.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:15:00 | 533.03 | 519.20 | 504.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-02-05 12:15:00 | 558.41 | 547.85 | 532.11 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 44 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 524.60 | 547.29 | 547.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 514.55 | 532.20 | 538.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 10:15:00 | 497.20 | 496.12 | 506.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 11:00:00 | 497.20 | 496.12 | 506.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 510.80 | 500.18 | 506.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 510.80 | 500.18 | 506.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 521.60 | 504.46 | 507.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 521.60 | 504.46 | 507.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 525.00 | 512.27 | 511.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 10:15:00 | 535.50 | 525.55 | 519.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 517.55 | 528.38 | 523.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 14:15:00 | 517.55 | 528.38 | 523.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 517.55 | 528.38 | 523.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 517.55 | 528.38 | 523.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 519.40 | 526.58 | 523.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 535.10 | 526.58 | 523.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 529.70 | 537.15 | 537.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 529.70 | 537.15 | 537.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 10:15:00 | 525.35 | 530.16 | 533.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 09:15:00 | 526.80 | 526.68 | 529.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-26 09:30:00 | 528.65 | 526.68 | 529.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 521.55 | 524.32 | 527.46 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 534.35 | 528.31 | 528.04 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 518.55 | 528.69 | 528.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 512.85 | 525.52 | 527.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 14:15:00 | 529.15 | 524.88 | 526.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 14:15:00 | 529.15 | 524.88 | 526.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 529.15 | 524.88 | 526.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 15:00:00 | 529.15 | 524.88 | 526.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 529.00 | 525.71 | 526.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 521.65 | 525.71 | 526.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 09:45:00 | 525.75 | 522.30 | 523.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 14:15:00 | 527.65 | 524.29 | 524.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 527.65 | 524.29 | 524.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 529.95 | 526.31 | 525.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 525.75 | 526.58 | 525.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 525.75 | 526.58 | 525.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 525.75 | 526.58 | 525.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 525.75 | 526.58 | 525.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 525.60 | 526.38 | 525.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 13:45:00 | 531.40 | 526.26 | 525.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 532.00 | 526.87 | 526.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:15:00 | 528.95 | 526.51 | 525.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:45:00 | 530.85 | 527.29 | 526.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 528.05 | 527.45 | 526.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:45:00 | 527.00 | 527.45 | 526.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 530.00 | 528.30 | 527.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 534.60 | 528.30 | 527.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 522.65 | 527.17 | 526.68 | SL hit (close<static) qty=1.00 sl=524.30 alert=retest2 |

### Cycle 50 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 518.50 | 525.39 | 525.95 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 13:15:00 | 535.55 | 527.59 | 526.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 14:15:00 | 536.90 | 529.45 | 527.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 11:15:00 | 544.65 | 545.46 | 539.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 11:45:00 | 546.90 | 545.46 | 539.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 529.80 | 543.87 | 541.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 529.80 | 543.87 | 541.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 526.30 | 540.35 | 540.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:00:00 | 526.30 | 540.35 | 540.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 509.65 | 534.21 | 537.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 496.85 | 515.28 | 524.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 09:15:00 | 497.80 | 497.06 | 505.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 13:15:00 | 507.50 | 497.71 | 503.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 507.50 | 497.71 | 503.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:00:00 | 507.50 | 497.71 | 503.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 514.15 | 501.00 | 504.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 514.15 | 501.00 | 504.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 504.60 | 501.72 | 504.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 503.05 | 501.72 | 504.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 11:00:00 | 501.45 | 501.25 | 503.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:30:00 | 500.55 | 501.21 | 502.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 477.90 | 485.86 | 491.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 476.38 | 485.86 | 491.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 475.52 | 485.86 | 491.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 479.00 | 476.72 | 483.77 | SL hit (close>ema200) qty=0.50 sl=476.72 alert=retest2 |

### Cycle 53 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 493.15 | 487.02 | 486.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 496.35 | 490.73 | 488.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 13:15:00 | 506.45 | 510.20 | 503.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 14:00:00 | 506.45 | 510.20 | 503.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 499.95 | 508.15 | 503.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 499.95 | 508.15 | 503.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 501.00 | 506.72 | 503.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 508.00 | 506.72 | 503.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 523.50 | 532.72 | 533.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 523.50 | 532.72 | 533.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 521.70 | 526.60 | 528.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 520.20 | 518.87 | 521.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 520.20 | 518.87 | 521.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 521.10 | 519.56 | 521.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 526.85 | 519.56 | 521.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 518.25 | 519.30 | 521.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 525.40 | 519.30 | 521.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 512.00 | 509.90 | 513.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 15:00:00 | 512.00 | 509.90 | 513.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 518.85 | 511.70 | 513.46 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 524.00 | 515.79 | 515.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 527.70 | 521.06 | 518.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 13:15:00 | 522.05 | 522.67 | 520.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 14:00:00 | 522.05 | 522.67 | 520.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 516.85 | 521.50 | 519.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 516.85 | 521.50 | 519.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 512.00 | 519.60 | 519.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 519.25 | 519.60 | 519.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 10:30:00 | 517.70 | 519.01 | 518.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 11:00:00 | 517.55 | 519.01 | 518.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 515.75 | 518.36 | 518.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 515.75 | 518.36 | 518.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 12:15:00 | 513.55 | 517.40 | 518.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 521.20 | 515.81 | 516.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 521.20 | 515.81 | 516.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 521.20 | 515.81 | 516.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 521.20 | 515.81 | 516.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 518.00 | 516.25 | 517.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 11:45:00 | 514.85 | 516.13 | 516.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 13:15:00 | 528.05 | 518.89 | 518.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 13:15:00 | 528.05 | 518.89 | 518.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 09:15:00 | 534.20 | 527.11 | 523.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 15:15:00 | 547.20 | 548.75 | 542.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 09:15:00 | 549.50 | 548.75 | 542.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 533.90 | 546.72 | 545.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:00:00 | 533.90 | 546.72 | 545.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 537.90 | 544.96 | 545.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 527.75 | 535.66 | 539.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 537.85 | 536.10 | 539.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 15:15:00 | 537.85 | 536.10 | 539.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 537.85 | 536.10 | 539.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 540.70 | 536.10 | 539.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 545.10 | 537.90 | 539.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 10:30:00 | 532.10 | 536.08 | 538.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 532.70 | 530.20 | 531.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 531.55 | 530.20 | 531.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 530.45 | 531.73 | 532.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 505.50 | 513.56 | 519.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 506.06 | 513.56 | 519.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 504.97 | 513.56 | 519.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 503.93 | 513.56 | 519.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 513.15 | 512.29 | 516.63 | SL hit (close>ema200) qty=0.50 sl=512.29 alert=retest2 |

### Cycle 59 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 527.25 | 516.12 | 515.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 534.30 | 519.76 | 517.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 528.35 | 530.42 | 525.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:30:00 | 530.55 | 530.42 | 525.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 539.95 | 539.83 | 536.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:15:00 | 543.00 | 539.62 | 536.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 565.70 | 572.51 | 572.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 565.70 | 572.51 | 572.62 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 581.40 | 573.64 | 572.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 588.15 | 578.42 | 575.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 578.70 | 582.47 | 579.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 578.70 | 582.47 | 579.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 578.70 | 582.47 | 579.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 578.70 | 582.47 | 579.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 577.10 | 581.40 | 579.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 573.20 | 581.40 | 579.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 567.95 | 578.71 | 578.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 567.95 | 578.71 | 578.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 570.75 | 577.12 | 577.38 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 597.50 | 577.18 | 576.58 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 535.65 | 580.29 | 582.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 513.65 | 566.96 | 575.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 524.95 | 517.83 | 536.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 524.95 | 517.83 | 536.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 545.30 | 524.15 | 536.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 545.50 | 524.15 | 536.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 551.85 | 529.69 | 537.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 551.55 | 529.69 | 537.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 541.75 | 536.78 | 539.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 541.75 | 536.78 | 539.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 545.95 | 538.62 | 539.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 532.55 | 538.62 | 539.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 11:30:00 | 539.45 | 537.24 | 538.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 538.45 | 536.26 | 536.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 14:15:00 | 538.45 | 536.26 | 536.03 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 533.30 | 535.72 | 535.92 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 14:15:00 | 538.55 | 536.19 | 536.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 541.95 | 537.47 | 536.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 538.25 | 539.06 | 537.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 13:15:00 | 538.25 | 539.06 | 537.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 538.25 | 539.06 | 537.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 538.25 | 539.06 | 537.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 541.30 | 539.51 | 538.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 542.55 | 540.59 | 539.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:15:00 | 543.75 | 540.59 | 539.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:15:00 | 543.25 | 540.99 | 540.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 545.10 | 542.77 | 541.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 545.00 | 543.21 | 541.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 547.95 | 544.46 | 543.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:45:00 | 547.00 | 545.01 | 543.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 541.20 | 544.59 | 544.39 | SL hit (close<static) qty=1.00 sl=541.70 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 541.20 | 543.92 | 544.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 540.85 | 542.85 | 543.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 537.50 | 537.43 | 540.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 09:30:00 | 536.70 | 537.43 | 540.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 540.10 | 537.96 | 540.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 540.10 | 537.96 | 540.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 539.25 | 538.22 | 540.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 540.95 | 538.22 | 540.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 540.00 | 538.57 | 540.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 540.10 | 538.57 | 540.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 540.10 | 538.88 | 540.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 539.35 | 538.88 | 540.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 540.05 | 539.11 | 540.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:45:00 | 540.80 | 539.11 | 540.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 540.00 | 539.29 | 540.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 537.75 | 539.29 | 540.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 540.35 | 539.50 | 540.08 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 541.20 | 540.33 | 540.32 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 15:15:00 | 536.25 | 539.52 | 539.95 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 549.45 | 541.50 | 540.81 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 540.40 | 543.34 | 543.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 536.15 | 541.90 | 542.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 537.35 | 536.71 | 538.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 537.35 | 536.71 | 538.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 537.35 | 536.71 | 538.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 535.40 | 536.71 | 538.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 536.00 | 536.62 | 537.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 532.00 | 536.11 | 537.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 543.55 | 537.60 | 538.11 | SL hit (close>static) qty=1.00 sl=540.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 540.45 | 535.91 | 535.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 13:15:00 | 543.55 | 540.09 | 538.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 575.30 | 581.69 | 574.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 575.30 | 581.69 | 574.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 575.30 | 581.69 | 574.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 574.95 | 581.69 | 574.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 577.45 | 580.84 | 575.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 575.45 | 580.84 | 575.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 575.60 | 579.79 | 575.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 575.60 | 579.79 | 575.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 570.35 | 577.90 | 574.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 570.30 | 577.90 | 574.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 569.00 | 576.12 | 574.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 568.10 | 576.12 | 574.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 557.00 | 570.84 | 572.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 556.10 | 567.89 | 570.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 564.35 | 562.97 | 566.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 564.35 | 562.97 | 566.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 564.35 | 562.97 | 566.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 565.45 | 562.97 | 566.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 573.45 | 564.72 | 566.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 573.35 | 564.72 | 566.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 575.00 | 566.78 | 567.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 575.00 | 566.78 | 567.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 569.80 | 567.38 | 567.33 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 565.25 | 567.29 | 567.31 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 567.75 | 567.38 | 567.35 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 558.80 | 565.92 | 566.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 13:15:00 | 553.15 | 563.37 | 565.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 563.30 | 560.26 | 563.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 563.30 | 560.26 | 563.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 563.30 | 560.26 | 563.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 563.30 | 560.26 | 563.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 565.55 | 561.32 | 563.46 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 570.00 | 565.17 | 564.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 579.10 | 573.71 | 569.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 15:15:00 | 586.90 | 587.18 | 581.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:15:00 | 592.00 | 587.18 | 581.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 595.95 | 602.02 | 598.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 595.95 | 602.02 | 598.23 | SL hit (close<ema400) qty=1.00 sl=598.23 alert=retest1 |

### Cycle 80 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 591.20 | 596.34 | 596.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 573.40 | 590.21 | 593.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 568.30 | 565.74 | 572.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 12:00:00 | 568.30 | 565.74 | 572.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 564.60 | 559.38 | 563.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:45:00 | 563.05 | 559.38 | 563.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 571.95 | 561.90 | 563.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 572.40 | 561.90 | 563.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 15:15:00 | 577.70 | 567.67 | 566.41 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 559.65 | 566.50 | 566.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 15:15:00 | 556.90 | 564.58 | 565.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 550.30 | 545.19 | 551.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 15:00:00 | 550.30 | 545.19 | 551.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 546.55 | 545.46 | 551.09 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 559.25 | 553.18 | 552.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 12:15:00 | 561.55 | 555.03 | 553.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 555.30 | 556.86 | 555.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 555.30 | 556.86 | 555.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 555.30 | 556.86 | 555.20 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 552.35 | 554.51 | 554.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 551.00 | 553.73 | 554.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 11:15:00 | 554.45 | 551.35 | 552.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 11:15:00 | 554.45 | 551.35 | 552.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 554.45 | 551.35 | 552.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 554.45 | 551.35 | 552.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 551.80 | 551.44 | 552.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 551.80 | 551.44 | 552.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 551.65 | 550.01 | 551.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 548.20 | 549.55 | 550.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 15:15:00 | 547.00 | 549.55 | 550.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:15:00 | 547.65 | 550.03 | 550.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 568.95 | 553.81 | 552.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 568.95 | 553.81 | 552.05 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 554.75 | 558.30 | 558.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 11:15:00 | 551.00 | 556.25 | 557.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 14:15:00 | 543.00 | 534.20 | 539.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 543.00 | 534.20 | 539.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 543.00 | 534.20 | 539.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 543.00 | 534.20 | 539.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 558.70 | 539.10 | 541.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:30:00 | 540.10 | 540.12 | 541.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 513.10 | 528.57 | 534.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 522.10 | 521.48 | 525.69 | SL hit (close>ema200) qty=0.50 sl=521.48 alert=retest2 |

### Cycle 87 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 520.00 | 519.35 | 519.33 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 516.30 | 519.08 | 519.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 515.70 | 518.06 | 518.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 520.20 | 518.07 | 518.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 520.20 | 518.07 | 518.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 520.20 | 518.07 | 518.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:45:00 | 520.60 | 518.07 | 518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 519.65 | 518.38 | 518.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 519.65 | 518.38 | 518.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 518.00 | 518.31 | 518.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 521.00 | 518.31 | 518.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 519.15 | 518.48 | 518.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:15:00 | 517.30 | 518.48 | 518.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 522.55 | 512.55 | 512.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 522.55 | 512.55 | 512.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 535.05 | 517.05 | 514.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 528.90 | 529.03 | 523.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 11:00:00 | 528.90 | 529.03 | 523.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 521.60 | 527.05 | 525.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 521.60 | 527.05 | 525.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 520.10 | 525.66 | 524.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 520.30 | 525.66 | 524.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 524.90 | 524.67 | 524.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:30:00 | 524.95 | 524.67 | 524.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 524.80 | 524.70 | 524.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 521.85 | 524.70 | 524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 523.60 | 524.48 | 524.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 521.65 | 524.48 | 524.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 524.70 | 524.52 | 524.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:45:00 | 532.45 | 527.20 | 525.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 532.55 | 529.10 | 526.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:00:00 | 532.00 | 533.20 | 530.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:30:00 | 532.10 | 532.96 | 530.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 528.15 | 532.00 | 530.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 528.15 | 532.00 | 530.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 527.75 | 531.15 | 530.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 524.85 | 529.89 | 529.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 524.85 | 529.89 | 529.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 15:15:00 | 522.35 | 528.38 | 529.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 529.20 | 527.91 | 528.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 12:15:00 | 529.20 | 527.91 | 528.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 529.20 | 527.91 | 528.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:30:00 | 528.80 | 527.91 | 528.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 527.55 | 527.84 | 528.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:15:00 | 525.75 | 527.84 | 528.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:45:00 | 526.20 | 527.81 | 528.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 15:15:00 | 526.20 | 527.81 | 528.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 534.00 | 522.62 | 524.26 | SL hit (close>static) qty=1.00 sl=529.80 alert=retest2 |

### Cycle 91 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 527.35 | 522.73 | 522.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 533.25 | 526.57 | 524.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 11:15:00 | 524.80 | 526.83 | 524.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 11:15:00 | 524.80 | 526.83 | 524.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 524.80 | 526.83 | 524.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:00:00 | 524.80 | 526.83 | 524.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 523.70 | 526.20 | 524.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:00:00 | 523.70 | 526.20 | 524.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 525.35 | 526.03 | 524.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:45:00 | 523.45 | 526.03 | 524.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 527.90 | 526.41 | 525.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:30:00 | 525.50 | 526.41 | 525.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 528.80 | 527.30 | 525.73 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 15:15:00 | 523.10 | 525.12 | 525.34 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 525.70 | 525.11 | 525.07 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 520.40 | 524.16 | 524.64 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 526.00 | 524.91 | 524.85 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 524.15 | 524.76 | 524.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 12:15:00 | 518.50 | 523.51 | 524.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 514.85 | 514.82 | 517.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 514.85 | 514.82 | 517.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 514.85 | 514.82 | 517.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 515.55 | 514.82 | 517.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 520.35 | 516.12 | 517.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 519.60 | 516.12 | 517.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 517.10 | 516.32 | 517.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 515.95 | 516.32 | 517.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 516.50 | 517.82 | 518.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 515.85 | 517.43 | 517.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 490.15 | 499.20 | 502.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 490.67 | 499.20 | 502.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 490.06 | 499.20 | 502.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 498.85 | 496.68 | 499.83 | SL hit (close>ema200) qty=0.50 sl=496.68 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 513.50 | 501.65 | 501.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 552.20 | 513.62 | 506.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 582.25 | 584.30 | 568.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 582.25 | 584.30 | 568.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 579.55 | 586.92 | 578.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 579.55 | 586.92 | 578.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 571.45 | 583.23 | 578.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 570.80 | 583.23 | 578.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 569.10 | 580.41 | 577.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:30:00 | 567.85 | 580.41 | 577.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 572.00 | 574.89 | 575.21 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 577.85 | 575.48 | 575.45 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 573.95 | 576.36 | 576.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 11:15:00 | 572.70 | 575.63 | 576.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 576.00 | 574.92 | 575.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 576.00 | 574.92 | 575.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 576.00 | 574.92 | 575.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:30:00 | 576.50 | 574.92 | 575.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 579.00 | 575.74 | 575.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 584.00 | 575.74 | 575.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 576.85 | 575.96 | 575.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 580.70 | 575.96 | 575.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 10:15:00 | 579.25 | 576.62 | 576.27 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 567.70 | 574.87 | 575.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 559.20 | 571.74 | 574.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 567.85 | 563.97 | 568.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 567.85 | 563.97 | 568.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 567.85 | 563.97 | 568.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:45:00 | 568.00 | 563.97 | 568.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 568.25 | 564.82 | 568.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 564.00 | 566.42 | 567.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 562.90 | 566.58 | 567.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:45:00 | 564.40 | 566.23 | 567.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 563.15 | 566.23 | 567.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 562.90 | 565.56 | 566.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 562.90 | 565.56 | 566.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 536.18 | 547.89 | 554.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 545.95 | 545.90 | 552.10 | SL hit (close>ema200) qty=0.50 sl=545.90 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 531.95 | 529.52 | 529.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 570.05 | 538.10 | 533.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 560.85 | 561.03 | 554.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 558.35 | 561.03 | 554.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 572.55 | 562.56 | 558.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 578.10 | 565.78 | 561.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 582.00 | 567.83 | 563.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:00:00 | 580.50 | 574.10 | 570.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:30:00 | 577.15 | 580.67 | 576.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 598.85 | 595.01 | 590.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 598.85 | 595.01 | 590.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 591.05 | 594.87 | 591.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 591.05 | 594.87 | 591.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 599.25 | 595.74 | 591.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 592.15 | 595.48 | 595.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 592.15 | 595.48 | 595.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 590.35 | 594.45 | 595.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 572.80 | 571.88 | 578.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 572.80 | 571.88 | 578.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 585.05 | 575.07 | 578.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 585.05 | 575.07 | 578.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 582.70 | 576.60 | 578.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 580.25 | 577.33 | 578.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:00:00 | 582.55 | 578.37 | 579.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 582.00 | 580.00 | 579.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 582.00 | 580.00 | 579.77 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 575.00 | 578.77 | 579.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 569.00 | 576.82 | 578.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 540.55 | 539.04 | 542.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 14:00:00 | 540.55 | 539.04 | 542.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 548.85 | 541.30 | 542.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 548.85 | 541.30 | 542.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 547.55 | 542.55 | 543.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:15:00 | 541.75 | 542.69 | 543.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 545.40 | 543.91 | 543.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 545.40 | 543.91 | 543.77 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 539.20 | 542.97 | 543.36 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 545.75 | 543.59 | 543.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 15:15:00 | 548.00 | 544.47 | 543.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 539.60 | 545.93 | 545.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 12:15:00 | 539.60 | 545.93 | 545.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 539.60 | 545.93 | 545.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 539.60 | 545.93 | 545.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 534.85 | 543.72 | 544.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 529.55 | 540.88 | 542.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 13:15:00 | 527.00 | 520.06 | 524.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 13:15:00 | 527.00 | 520.06 | 524.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 527.00 | 520.06 | 524.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 525.65 | 520.06 | 524.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 523.90 | 520.83 | 524.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 513.00 | 524.69 | 525.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 11:15:00 | 487.35 | 497.03 | 502.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 485.35 | 482.44 | 489.07 | SL hit (close>ema200) qty=0.50 sl=482.44 alert=retest2 |

### Cycle 111 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 503.75 | 493.02 | 492.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 505.90 | 497.27 | 494.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 501.45 | 503.30 | 499.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 14:00:00 | 501.45 | 503.30 | 499.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 525.20 | 528.24 | 524.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 525.55 | 528.24 | 524.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 526.15 | 527.83 | 525.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 524.35 | 527.83 | 525.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 530.65 | 528.39 | 525.52 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 518.90 | 524.23 | 524.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 516.15 | 521.51 | 523.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 521.90 | 520.62 | 522.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 521.90 | 520.62 | 522.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 521.90 | 520.62 | 522.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 521.90 | 520.62 | 522.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 522.85 | 521.06 | 522.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 516.30 | 521.06 | 522.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 525.20 | 521.89 | 522.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 525.20 | 521.89 | 522.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 523.10 | 522.13 | 522.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 519.30 | 521.89 | 522.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 521.80 | 521.93 | 522.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:45:00 | 521.20 | 521.30 | 522.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 493.33 | 504.73 | 511.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 495.71 | 504.73 | 511.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 495.14 | 504.73 | 511.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 12:15:00 | 500.50 | 499.71 | 507.03 | SL hit (close>ema200) qty=0.50 sl=499.71 alert=retest2 |

### Cycle 113 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 521.00 | 508.73 | 507.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 525.50 | 513.70 | 510.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 550.50 | 551.06 | 543.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 534.70 | 548.53 | 544.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 534.70 | 548.53 | 544.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 534.70 | 548.53 | 544.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 541.30 | 547.09 | 544.19 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 521.00 | 538.54 | 540.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 514.80 | 530.48 | 536.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 532.15 | 525.83 | 531.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 532.15 | 525.83 | 531.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 532.15 | 525.83 | 531.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 539.30 | 525.83 | 531.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 530.00 | 526.66 | 531.15 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 540.35 | 533.40 | 533.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 548.90 | 537.63 | 535.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 536.70 | 539.14 | 536.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 13:15:00 | 536.70 | 539.14 | 536.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 536.70 | 539.14 | 536.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:00:00 | 536.70 | 539.14 | 536.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 539.45 | 539.20 | 537.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 15:15:00 | 542.00 | 539.20 | 537.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 541.60 | 544.77 | 543.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 530.45 | 541.65 | 542.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 530.45 | 541.65 | 542.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 527.90 | 538.90 | 540.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 518.05 | 514.80 | 521.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 518.75 | 514.80 | 521.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 522.75 | 516.39 | 521.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 522.75 | 516.39 | 521.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 517.60 | 516.63 | 521.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:30:00 | 515.55 | 517.28 | 521.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 10:15:00 | 523.50 | 519.38 | 521.22 | SL hit (close>static) qty=1.00 sl=522.90 alert=retest2 |

### Cycle 117 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 515.15 | 512.58 | 512.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 518.55 | 513.78 | 513.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 524.20 | 524.76 | 521.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 524.20 | 524.76 | 521.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 521.75 | 524.81 | 522.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 521.75 | 524.81 | 522.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 519.45 | 523.74 | 522.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 519.45 | 523.74 | 522.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 515.95 | 520.48 | 520.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 10:15:00 | 513.90 | 517.22 | 519.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 521.95 | 518.16 | 519.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 521.95 | 518.16 | 519.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 521.95 | 518.16 | 519.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 521.65 | 518.16 | 519.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 526.05 | 519.74 | 519.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 526.05 | 519.74 | 519.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 525.70 | 520.93 | 520.48 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 517.45 | 521.46 | 521.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 511.10 | 519.51 | 520.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 505.10 | 503.23 | 507.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 505.10 | 503.23 | 507.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 505.10 | 503.23 | 507.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 508.50 | 503.23 | 507.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 504.15 | 503.41 | 507.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 503.30 | 503.29 | 507.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 503.85 | 503.40 | 506.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 500.10 | 503.56 | 506.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:15:00 | 503.90 | 503.56 | 506.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 515.90 | 506.31 | 507.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 515.90 | 506.31 | 507.16 | SL hit (close>static) qty=1.00 sl=507.90 alert=retest2 |

### Cycle 121 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 513.25 | 508.80 | 508.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 522.05 | 512.05 | 509.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 528.40 | 528.48 | 523.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:15:00 | 529.95 | 528.48 | 523.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 519.50 | 526.44 | 524.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 519.50 | 526.44 | 524.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 518.30 | 524.81 | 523.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 517.90 | 524.81 | 523.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 509.20 | 520.24 | 521.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 506.50 | 517.49 | 520.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 501.90 | 494.41 | 499.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 501.90 | 494.41 | 499.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 501.90 | 494.41 | 499.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 501.90 | 494.41 | 499.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 500.30 | 495.58 | 499.41 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 509.50 | 502.18 | 501.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 510.35 | 503.81 | 502.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 545.90 | 546.33 | 540.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:30:00 | 547.60 | 546.33 | 540.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 541.20 | 544.59 | 541.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 541.40 | 544.59 | 541.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 544.65 | 544.60 | 541.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 548.55 | 544.13 | 541.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 546.55 | 546.13 | 543.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 546.85 | 545.39 | 544.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 547.10 | 545.25 | 544.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 545.15 | 547.13 | 545.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:30:00 | 544.30 | 547.13 | 545.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 542.70 | 546.24 | 545.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:45:00 | 541.35 | 546.24 | 545.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 14:15:00 | 540.90 | 544.08 | 544.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 540.90 | 544.08 | 544.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 535.35 | 541.76 | 543.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 543.95 | 533.34 | 535.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 543.95 | 533.34 | 535.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 543.95 | 533.34 | 535.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 543.95 | 533.34 | 535.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 545.80 | 535.84 | 536.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 546.90 | 535.84 | 536.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 547.45 | 538.16 | 537.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 550.65 | 542.02 | 539.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 545.05 | 546.46 | 543.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 13:15:00 | 545.05 | 546.46 | 543.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 545.05 | 546.46 | 543.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:00:00 | 545.05 | 546.46 | 543.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 547.00 | 546.57 | 543.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 541.70 | 546.57 | 543.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 526.60 | 542.52 | 542.40 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 533.25 | 540.66 | 541.57 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 547.50 | 541.30 | 540.61 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 527.30 | 539.22 | 540.26 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 536.95 | 536.04 | 535.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 539.90 | 536.81 | 536.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 13:15:00 | 577.65 | 578.02 | 568.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 13:45:00 | 578.65 | 578.02 | 568.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 577.20 | 583.01 | 578.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 577.20 | 583.01 | 578.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 578.65 | 582.13 | 578.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 574.00 | 582.13 | 578.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 578.55 | 581.42 | 578.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 578.55 | 581.42 | 578.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 579.20 | 580.97 | 578.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 578.95 | 580.97 | 578.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 582.70 | 581.32 | 578.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 578.55 | 581.32 | 578.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 579.15 | 580.77 | 579.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 579.35 | 580.77 | 579.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 576.05 | 579.83 | 578.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 575.10 | 579.83 | 578.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 576.50 | 579.16 | 578.69 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 574.20 | 578.17 | 578.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 565.40 | 574.23 | 576.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 577.55 | 572.53 | 574.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 577.55 | 572.53 | 574.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 577.55 | 572.53 | 574.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 577.55 | 572.53 | 574.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 569.25 | 571.87 | 574.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 566.95 | 571.87 | 574.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 12:15:00 | 587.50 | 576.68 | 575.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 587.50 | 576.68 | 575.55 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 571.05 | 578.21 | 578.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 566.30 | 572.73 | 575.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 571.10 | 570.82 | 573.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 571.10 | 570.82 | 573.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 571.10 | 570.82 | 573.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 571.10 | 570.82 | 573.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 574.50 | 564.29 | 568.07 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 582.50 | 570.03 | 569.91 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 563.25 | 570.83 | 570.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 561.95 | 569.05 | 570.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 559.00 | 558.92 | 562.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 557.50 | 558.92 | 562.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 566.00 | 560.34 | 562.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 566.00 | 560.34 | 562.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 565.25 | 561.32 | 562.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:00:00 | 561.10 | 563.12 | 563.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 570.35 | 560.17 | 560.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 570.35 | 560.17 | 560.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 573.45 | 562.83 | 561.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 578.90 | 579.66 | 574.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:45:00 | 579.40 | 579.66 | 574.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 600.60 | 605.30 | 600.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 601.15 | 605.30 | 600.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 608.85 | 606.01 | 601.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 612.00 | 606.01 | 601.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:00:00 | 611.20 | 607.84 | 602.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 596.30 | 605.30 | 603.28 | SL hit (close<static) qty=1.00 sl=600.40 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 594.95 | 601.28 | 601.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 593.10 | 599.64 | 601.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 595.25 | 594.41 | 597.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 595.25 | 594.41 | 597.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 595.25 | 594.41 | 597.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 595.25 | 594.41 | 597.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 595.80 | 594.12 | 595.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 597.70 | 594.12 | 595.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 592.45 | 593.78 | 595.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 596.65 | 593.78 | 595.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 600.10 | 595.05 | 595.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 600.10 | 595.05 | 595.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 597.15 | 595.47 | 596.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 595.40 | 595.47 | 596.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 595.30 | 593.83 | 594.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 599.50 | 595.20 | 594.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 599.50 | 595.20 | 594.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 605.15 | 598.37 | 596.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 603.00 | 603.25 | 599.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 603.00 | 603.25 | 599.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 603.00 | 603.25 | 599.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 601.30 | 603.25 | 599.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 598.85 | 602.37 | 599.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 598.85 | 602.37 | 599.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 597.70 | 601.43 | 599.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 597.70 | 601.43 | 599.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 599.80 | 600.89 | 599.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:00:00 | 601.45 | 601.00 | 599.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 602.45 | 601.43 | 600.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 629.00 | 631.32 | 631.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 629.00 | 631.32 | 631.45 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 637.35 | 631.80 | 631.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 648.60 | 636.25 | 633.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 627.20 | 642.94 | 639.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 627.20 | 642.94 | 639.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 627.20 | 642.94 | 639.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 627.20 | 642.94 | 639.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 624.50 | 636.93 | 637.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 619.70 | 627.72 | 631.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 627.05 | 626.96 | 630.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 627.05 | 626.96 | 630.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 624.45 | 625.03 | 627.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:30:00 | 626.60 | 625.03 | 627.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 623.45 | 624.72 | 626.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 627.15 | 624.72 | 626.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 626.70 | 625.05 | 626.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 626.70 | 625.05 | 626.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 630.65 | 626.17 | 626.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 628.50 | 626.17 | 626.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 631.00 | 627.14 | 627.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 634.40 | 629.41 | 628.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 632.95 | 633.48 | 631.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 632.95 | 633.48 | 631.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 632.95 | 633.48 | 631.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 631.60 | 633.48 | 631.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 633.95 | 633.42 | 631.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 631.90 | 633.42 | 631.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 632.60 | 633.26 | 631.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 631.10 | 633.26 | 631.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 630.60 | 632.73 | 631.49 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 626.10 | 630.35 | 630.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 615.85 | 625.87 | 628.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 618.35 | 618.07 | 622.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 618.35 | 618.07 | 622.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 618.35 | 618.07 | 622.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 615.00 | 617.64 | 620.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 627.10 | 619.25 | 619.47 | SL hit (close>static) qty=1.00 sl=622.50 alert=retest2 |

### Cycle 143 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 630.00 | 621.40 | 620.42 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 620.50 | 621.47 | 621.52 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 625.05 | 622.16 | 621.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 630.40 | 623.81 | 622.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 626.25 | 631.00 | 628.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 626.25 | 631.00 | 628.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 626.25 | 631.00 | 628.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 626.25 | 631.00 | 628.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 628.40 | 630.48 | 628.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 634.50 | 630.48 | 628.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 643.85 | 648.39 | 648.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 643.85 | 648.39 | 648.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 641.95 | 647.10 | 648.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 638.95 | 638.91 | 642.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 638.55 | 638.91 | 642.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 641.30 | 639.43 | 641.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 641.25 | 639.43 | 641.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 639.00 | 639.34 | 641.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 636.75 | 638.49 | 640.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 642.70 | 631.81 | 632.98 | SL hit (close>static) qty=1.00 sl=641.60 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 635.10 | 633.71 | 633.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 639.40 | 634.90 | 634.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 635.25 | 636.57 | 635.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 12:15:00 | 635.25 | 636.57 | 635.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 635.25 | 636.57 | 635.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 635.25 | 636.57 | 635.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 634.45 | 636.15 | 635.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 633.60 | 636.15 | 635.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 634.50 | 635.82 | 635.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 632.80 | 635.82 | 635.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 635.10 | 640.04 | 638.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 631.60 | 640.04 | 638.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 637.00 | 639.43 | 638.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 632.55 | 639.43 | 638.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 634.50 | 637.64 | 637.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:15:00 | 635.00 | 637.64 | 637.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 636.35 | 637.38 | 637.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 11:15:00 | 631.35 | 635.37 | 636.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 634.10 | 632.71 | 634.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 634.10 | 632.71 | 634.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 634.10 | 632.71 | 634.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 634.10 | 632.71 | 634.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 635.65 | 633.30 | 634.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 635.65 | 633.30 | 634.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 636.90 | 634.02 | 634.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 637.00 | 634.02 | 634.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 633.60 | 633.93 | 634.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 631.45 | 634.06 | 634.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 632.30 | 634.04 | 634.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 647.10 | 632.18 | 630.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 647.10 | 632.18 | 630.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 650.10 | 635.76 | 632.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 638.65 | 639.60 | 634.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:45:00 | 638.35 | 639.60 | 634.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 629.95 | 637.67 | 634.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 629.95 | 637.67 | 634.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 634.45 | 637.03 | 634.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:00:00 | 635.35 | 636.69 | 634.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:45:00 | 635.85 | 636.24 | 634.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:30:00 | 635.30 | 636.04 | 634.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 624.00 | 633.11 | 633.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 624.00 | 633.11 | 633.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 623.90 | 627.84 | 629.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 621.65 | 617.82 | 621.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 12:15:00 | 621.65 | 617.82 | 621.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 621.65 | 617.82 | 621.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 621.65 | 617.82 | 621.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 621.85 | 619.01 | 621.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 621.85 | 619.01 | 621.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 622.45 | 619.70 | 621.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 617.25 | 619.70 | 621.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 618.00 | 619.36 | 621.54 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 627.10 | 623.12 | 622.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 10:15:00 | 629.65 | 624.43 | 623.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 635.00 | 635.64 | 631.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 13:00:00 | 635.00 | 635.64 | 631.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 629.50 | 633.86 | 631.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 629.50 | 633.86 | 631.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 630.10 | 633.11 | 631.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 629.05 | 633.11 | 631.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 631.50 | 632.79 | 631.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 630.40 | 632.79 | 631.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 635.05 | 633.24 | 631.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 630.00 | 633.24 | 631.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 637.50 | 635.49 | 633.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 641.50 | 636.74 | 634.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 670.50 | 671.63 | 671.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 670.50 | 671.63 | 671.74 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 673.20 | 671.58 | 671.37 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 668.00 | 670.71 | 671.05 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 672.15 | 671.27 | 671.18 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 668.50 | 670.72 | 670.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 664.30 | 669.44 | 670.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 653.30 | 652.96 | 657.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 653.90 | 652.96 | 657.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 653.00 | 653.14 | 656.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 651.00 | 653.14 | 656.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 660.55 | 654.28 | 656.21 | SL hit (close>static) qty=1.00 sl=657.05 alert=retest2 |

### Cycle 157 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 662.80 | 658.33 | 657.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 666.35 | 659.94 | 658.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 665.50 | 666.62 | 663.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 665.50 | 666.62 | 663.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 664.35 | 666.17 | 663.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 664.35 | 666.17 | 663.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 664.20 | 665.77 | 663.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 664.20 | 665.77 | 663.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 664.05 | 665.43 | 663.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 668.20 | 665.43 | 663.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 665.40 | 666.86 | 666.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 661.55 | 665.35 | 665.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 661.55 | 665.35 | 665.67 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 675.45 | 666.77 | 666.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 680.95 | 669.61 | 667.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 669.60 | 673.17 | 670.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 669.60 | 673.17 | 670.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 669.60 | 673.17 | 670.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 669.60 | 673.17 | 670.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 672.80 | 673.09 | 670.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 674.00 | 672.96 | 670.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 674.00 | 673.28 | 671.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 674.50 | 673.52 | 671.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 677.80 | 671.50 | 670.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 685.35 | 674.27 | 672.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 688.85 | 674.27 | 672.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 698.10 | 701.86 | 702.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 698.10 | 701.86 | 702.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 697.75 | 701.04 | 701.95 | Break + close below crossover candle low |

### Cycle 161 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 717.60 | 702.55 | 702.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 718.05 | 711.85 | 710.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 733.85 | 739.98 | 732.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 12:00:00 | 733.85 | 739.98 | 732.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 736.05 | 739.20 | 732.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 739.15 | 738.78 | 732.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 748.55 | 738.17 | 733.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 765.50 | 772.12 | 772.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 765.50 | 772.12 | 772.42 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 775.80 | 772.63 | 772.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 776.85 | 773.48 | 772.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 773.60 | 774.08 | 773.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 773.60 | 774.08 | 773.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 773.60 | 774.08 | 773.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 773.60 | 774.08 | 773.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 774.85 | 774.24 | 773.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:30:00 | 772.70 | 774.24 | 773.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 775.50 | 774.49 | 773.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 778.70 | 774.49 | 773.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 769.75 | 773.54 | 773.27 | SL hit (close<static) qty=1.00 sl=773.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 766.90 | 772.21 | 772.69 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 805.80 | 778.53 | 775.45 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 772.30 | 774.23 | 774.36 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 780.45 | 775.47 | 774.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 14:15:00 | 782.25 | 776.83 | 775.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 822.00 | 824.13 | 814.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 822.00 | 824.13 | 814.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 816.30 | 822.27 | 816.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 816.30 | 822.27 | 816.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 819.90 | 821.80 | 816.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 815.70 | 821.80 | 816.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 872.70 | 876.78 | 871.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 872.70 | 876.78 | 871.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 871.25 | 875.67 | 871.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 10:45:00 | 875.40 | 875.24 | 872.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 864.45 | 873.09 | 871.63 | SL hit (close<static) qty=1.00 sl=868.70 alert=retest2 |

### Cycle 168 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 865.60 | 870.25 | 870.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 864.05 | 869.01 | 869.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 868.80 | 865.04 | 867.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 868.80 | 865.04 | 867.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 868.80 | 865.04 | 867.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 868.80 | 865.04 | 867.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 870.90 | 866.21 | 867.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 872.65 | 866.21 | 867.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 873.95 | 867.76 | 868.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 874.55 | 867.76 | 868.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 873.95 | 869.00 | 868.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 877.15 | 870.63 | 869.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 12:15:00 | 869.60 | 870.96 | 869.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 12:15:00 | 869.60 | 870.96 | 869.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 869.60 | 870.96 | 869.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 869.80 | 870.96 | 869.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 875.10 | 871.79 | 870.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 869.20 | 871.79 | 870.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 856.10 | 870.41 | 870.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 856.10 | 870.41 | 870.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 866.55 | 869.64 | 869.97 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 870.30 | 869.00 | 868.91 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 865.15 | 868.23 | 868.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 10:15:00 | 863.60 | 867.30 | 868.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 869.00 | 867.21 | 867.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 869.00 | 867.21 | 867.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 869.00 | 867.21 | 867.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 869.00 | 867.21 | 867.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 869.05 | 867.58 | 867.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 882.10 | 867.58 | 867.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 881.95 | 870.45 | 869.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 887.35 | 873.83 | 870.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 11:15:00 | 884.85 | 885.52 | 880.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 12:00:00 | 884.85 | 885.52 | 880.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 887.50 | 886.96 | 882.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 883.15 | 886.96 | 882.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 881.05 | 885.78 | 882.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 881.05 | 885.78 | 882.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 884.75 | 885.57 | 882.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 886.60 | 885.82 | 883.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 886.45 | 886.15 | 883.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 886.50 | 886.22 | 883.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 887.85 | 885.96 | 884.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 882.55 | 886.97 | 885.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 881.00 | 886.97 | 885.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 883.90 | 886.36 | 885.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 877.35 | 886.36 | 885.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 872.35 | 883.55 | 884.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 872.35 | 883.55 | 884.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 864.75 | 879.79 | 882.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 864.00 | 859.26 | 865.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 864.00 | 859.26 | 865.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 864.00 | 859.26 | 865.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 864.00 | 859.26 | 865.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 862.50 | 859.91 | 864.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 863.60 | 859.91 | 864.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 867.75 | 861.48 | 865.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 867.75 | 861.48 | 865.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 869.00 | 862.98 | 865.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 869.25 | 862.98 | 865.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 873.00 | 868.01 | 867.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 888.05 | 872.01 | 869.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 874.80 | 881.15 | 876.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 874.80 | 881.15 | 876.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 874.80 | 881.15 | 876.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 877.80 | 881.15 | 876.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 866.90 | 878.30 | 875.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 866.90 | 878.30 | 875.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 865.25 | 872.66 | 873.38 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 882.75 | 872.45 | 871.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 10:15:00 | 888.80 | 875.72 | 873.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 872.00 | 881.45 | 878.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 872.00 | 881.45 | 878.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 872.00 | 881.45 | 878.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 872.00 | 881.45 | 878.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 868.10 | 878.78 | 877.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 868.75 | 878.78 | 877.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 865.20 | 874.30 | 875.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 862.85 | 872.01 | 874.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 819.25 | 811.66 | 824.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 819.25 | 811.66 | 824.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 810.70 | 812.99 | 823.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 809.90 | 812.99 | 823.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 809.55 | 812.49 | 821.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 809.80 | 811.73 | 819.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 769.40 | 786.92 | 800.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 769.07 | 786.92 | 800.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 769.31 | 786.92 | 800.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 789.35 | 787.41 | 799.07 | SL hit (close>ema200) qty=0.50 sl=787.41 alert=retest2 |

### Cycle 179 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 780.80 | 777.39 | 777.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 782.60 | 778.43 | 777.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 784.55 | 785.71 | 783.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 784.55 | 785.71 | 783.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 784.55 | 785.71 | 783.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 783.00 | 785.71 | 783.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 784.80 | 785.53 | 783.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 784.05 | 785.53 | 783.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 783.50 | 784.81 | 783.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:30:00 | 780.40 | 784.81 | 783.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 781.70 | 784.18 | 783.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 781.05 | 784.18 | 783.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 778.75 | 783.10 | 782.69 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 774.75 | 781.43 | 781.97 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 788.55 | 780.68 | 779.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 789.65 | 783.76 | 781.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 827.55 | 831.56 | 819.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:00:00 | 827.55 | 831.56 | 819.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 857.20 | 860.03 | 855.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 856.05 | 860.03 | 855.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 854.25 | 858.88 | 855.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 853.95 | 858.88 | 855.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 852.90 | 857.68 | 855.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 852.90 | 857.68 | 855.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 851.65 | 856.47 | 854.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 851.65 | 856.47 | 854.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 865.05 | 857.95 | 855.75 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 836.40 | 852.97 | 854.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 833.00 | 848.97 | 852.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 845.25 | 839.32 | 845.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 845.25 | 839.32 | 845.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 845.25 | 839.32 | 845.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 848.00 | 839.32 | 845.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 855.15 | 842.49 | 846.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 855.15 | 842.49 | 846.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 844.40 | 842.87 | 845.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:15:00 | 843.80 | 842.87 | 845.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:00:00 | 840.10 | 842.31 | 845.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 843.00 | 826.27 | 825.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 843.00 | 826.27 | 825.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 852.00 | 836.86 | 831.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 854.40 | 855.27 | 850.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 12:30:00 | 851.50 | 855.27 | 850.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 845.35 | 853.28 | 849.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 845.35 | 853.28 | 849.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 846.80 | 851.99 | 849.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:15:00 | 847.00 | 851.99 | 849.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 847.00 | 850.99 | 849.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 853.05 | 850.99 | 849.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 840.00 | 848.61 | 848.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 840.00 | 848.61 | 848.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 842.25 | 847.34 | 847.90 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 851.50 | 848.22 | 848.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 867.30 | 852.04 | 849.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 884.05 | 886.38 | 874.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 884.05 | 886.38 | 874.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 874.00 | 882.32 | 874.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 873.80 | 882.32 | 874.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 875.00 | 880.86 | 874.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:15:00 | 865.00 | 880.86 | 874.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 862.20 | 877.13 | 873.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 862.20 | 877.13 | 873.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 865.15 | 874.73 | 872.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:15:00 | 860.60 | 874.73 | 872.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 860.50 | 869.37 | 870.33 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 883.00 | 872.88 | 871.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 883.30 | 876.76 | 873.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 896.70 | 908.56 | 903.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 896.70 | 908.56 | 903.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 896.70 | 908.56 | 903.22 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 875.60 | 900.15 | 900.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 864.70 | 889.14 | 894.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 839.70 | 836.94 | 856.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 866.65 | 836.94 | 856.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 860.55 | 841.66 | 856.98 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 873.35 | 864.81 | 863.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 881.15 | 869.60 | 866.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 873.20 | 875.99 | 872.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 873.20 | 875.99 | 872.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 873.20 | 875.99 | 872.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 873.20 | 875.99 | 872.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 870.60 | 874.91 | 872.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 867.65 | 874.91 | 872.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 867.25 | 873.38 | 872.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:15:00 | 863.70 | 873.38 | 872.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 864.75 | 871.66 | 871.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 865.75 | 871.66 | 871.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 870.50 | 871.42 | 871.43 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 872.00 | 871.50 | 871.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 900.45 | 877.29 | 874.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 905.30 | 905.94 | 896.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 905.30 | 905.94 | 896.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 886.55 | 902.39 | 896.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 878.65 | 902.39 | 896.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 897.00 | 901.31 | 896.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 901.25 | 901.24 | 897.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:30:00 | 901.40 | 900.47 | 897.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 885.10 | 896.29 | 896.17 | SL hit (close<static) qty=1.00 sl=885.40 alert=retest2 |

### Cycle 192 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 889.80 | 894.99 | 895.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 878.20 | 890.22 | 893.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 875.80 | 875.74 | 880.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 10:15:00 | 875.80 | 875.74 | 880.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 875.80 | 875.74 | 880.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 875.80 | 875.74 | 880.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 877.25 | 876.04 | 880.64 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 890.50 | 883.41 | 882.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 10:15:00 | 904.60 | 887.65 | 884.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 932.65 | 934.82 | 921.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 932.65 | 934.82 | 921.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 925.65 | 931.62 | 924.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 933.20 | 931.62 | 924.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 968.00 | 979.46 | 981.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 968.00 | 979.46 | 981.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 938.45 | 969.18 | 975.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 954.00 | 945.83 | 956.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 954.00 | 945.83 | 956.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 954.00 | 945.83 | 956.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 954.70 | 945.83 | 956.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 945.20 | 945.71 | 955.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 936.00 | 944.38 | 954.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 943.00 | 944.91 | 953.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 941.00 | 943.96 | 952.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 943.10 | 945.28 | 949.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 889.20 | 933.80 | 943.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 895.85 | 933.80 | 943.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 893.95 | 933.80 | 943.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 895.94 | 933.80 | 943.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 909.95 | 905.75 | 920.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 909.95 | 905.75 | 920.42 | SL hit (close>ema200) qty=0.50 sl=905.75 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 926.60 | 923.68 | 923.63 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 921.60 | 923.35 | 923.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 916.35 | 921.95 | 922.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 877.35 | 870.63 | 883.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 877.35 | 870.63 | 883.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 877.35 | 870.63 | 883.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 881.40 | 870.63 | 883.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 873.10 | 871.94 | 881.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:00:00 | 867.00 | 870.97 | 879.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 886.65 | 876.31 | 878.32 | SL hit (close>static) qty=1.00 sl=885.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 889.70 | 881.00 | 880.23 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 863.45 | 877.58 | 879.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 857.15 | 869.49 | 874.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 891.50 | 871.72 | 874.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 891.50 | 871.72 | 874.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 891.50 | 871.72 | 874.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 893.90 | 871.72 | 874.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 891.90 | 875.76 | 875.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 890.70 | 875.76 | 875.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 885.90 | 877.78 | 876.81 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 842.90 | 871.91 | 874.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 841.95 | 857.50 | 866.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 862.05 | 851.34 | 860.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 862.05 | 851.34 | 860.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 862.05 | 851.34 | 860.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 864.40 | 851.34 | 860.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 861.95 | 853.46 | 860.40 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 870.70 | 864.94 | 864.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 900.75 | 872.10 | 867.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 881.30 | 896.92 | 885.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 881.30 | 896.92 | 885.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 881.30 | 896.92 | 885.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 884.15 | 896.92 | 885.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 873.55 | 892.24 | 884.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 873.55 | 892.24 | 884.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 871.00 | 879.81 | 880.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 859.25 | 875.70 | 878.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 865.60 | 858.72 | 866.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 865.60 | 858.72 | 866.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 865.60 | 858.72 | 866.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 864.80 | 858.72 | 866.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 864.30 | 859.84 | 866.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 886.65 | 868.11 | 869.19 | SL hit (close>static) qty=1.00 sl=881.60 alert=retest2 |

### Cycle 203 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 886.40 | 871.76 | 870.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 890.05 | 875.42 | 872.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 849.80 | 871.85 | 871.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 849.80 | 871.85 | 871.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 849.80 | 871.85 | 871.49 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 852.40 | 867.96 | 869.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 11:15:00 | 846.05 | 863.58 | 867.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 868.00 | 862.68 | 866.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 868.00 | 862.68 | 866.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 868.00 | 862.68 | 866.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 868.00 | 862.68 | 866.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 872.10 | 864.56 | 866.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 872.10 | 864.56 | 866.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 870.00 | 865.65 | 867.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 890.45 | 865.65 | 867.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 890.05 | 870.53 | 869.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 905.80 | 895.13 | 887.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 945.35 | 948.04 | 930.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 945.35 | 948.04 | 930.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 942.45 | 955.02 | 946.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 950.00 | 953.87 | 946.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 952.15 | 953.77 | 953.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 939.75 | 950.96 | 952.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 939.75 | 950.96 | 952.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 10:15:00 | 934.20 | 947.61 | 950.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 942.95 | 941.52 | 945.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:15:00 | 943.60 | 941.52 | 945.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 934.65 | 940.14 | 944.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 932.20 | 937.92 | 940.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:30:00 | 931.75 | 936.37 | 939.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 932.00 | 934.45 | 938.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 11:15:00 | 930.80 | 934.52 | 937.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 928.90 | 927.29 | 931.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 930.60 | 927.29 | 931.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 927.50 | 927.33 | 931.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:30:00 | 926.70 | 927.18 | 930.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 925.85 | 927.34 | 930.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 921.25 | 911.49 | 910.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 921.25 | 911.49 | 910.84 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 899.95 | 909.18 | 909.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 869.30 | 901.21 | 906.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 843.50 | 835.68 | 848.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 843.50 | 835.68 | 848.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 851.25 | 839.72 | 847.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 851.25 | 839.72 | 847.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 849.90 | 841.75 | 848.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 868.45 | 841.75 | 848.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 862.05 | 845.81 | 849.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 857.05 | 848.06 | 850.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 865.70 | 853.62 | 852.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 865.70 | 853.62 | 852.16 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 854.70 | 856.18 | 856.36 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 864.85 | 857.92 | 857.13 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-19 09:15:00 | 285.70 | 2023-05-23 09:15:00 | 288.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-05-22 09:45:00 | 287.00 | 2023-05-23 09:15:00 | 288.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-06-06 09:30:00 | 279.70 | 2023-06-06 10:15:00 | 276.55 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-06-06 13:45:00 | 279.80 | 2023-06-12 09:15:00 | 277.70 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-07-07 12:15:00 | 306.10 | 2023-07-24 15:15:00 | 336.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-27 15:15:00 | 326.70 | 2023-07-28 09:15:00 | 340.40 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2023-08-07 09:15:00 | 345.95 | 2023-08-08 13:15:00 | 380.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-07 09:45:00 | 344.35 | 2023-08-08 13:15:00 | 378.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-07 10:30:00 | 346.30 | 2023-08-08 13:15:00 | 380.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-29 10:15:00 | 394.15 | 2023-09-05 09:15:00 | 393.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2023-09-11 09:30:00 | 392.05 | 2023-09-12 09:15:00 | 383.10 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-09-11 11:00:00 | 392.00 | 2023-09-12 09:15:00 | 383.10 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2023-09-11 12:15:00 | 391.00 | 2023-09-12 09:15:00 | 383.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2023-09-18 09:15:00 | 405.50 | 2023-09-27 10:15:00 | 446.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-05 09:15:00 | 439.35 | 2023-10-05 10:15:00 | 430.45 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-10-18 10:15:00 | 434.00 | 2023-10-18 14:15:00 | 415.65 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2023-10-18 11:30:00 | 432.00 | 2023-10-18 14:15:00 | 415.65 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2023-10-20 11:45:00 | 421.35 | 2023-10-26 09:15:00 | 400.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 420.00 | 2023-10-26 09:15:00 | 399.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:45:00 | 421.35 | 2023-10-27 09:15:00 | 411.80 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2023-10-23 09:15:00 | 420.00 | 2023-10-27 09:15:00 | 411.80 | STOP_HIT | 0.50 | 1.95% |
| BUY | retest2 | 2023-10-31 09:15:00 | 418.00 | 2023-11-06 14:15:00 | 420.80 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2023-10-31 10:00:00 | 417.65 | 2023-11-06 14:15:00 | 420.80 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2023-12-06 09:15:00 | 413.45 | 2023-12-15 14:15:00 | 454.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-06 10:00:00 | 415.05 | 2023-12-15 15:15:00 | 456.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-29 14:15:00 | 418.75 | 2024-01-02 09:15:00 | 417.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-01-02 09:30:00 | 420.20 | 2024-01-02 10:15:00 | 411.65 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-01-11 10:45:00 | 423.00 | 2024-01-11 11:15:00 | 427.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-01-15 15:00:00 | 435.05 | 2024-01-24 09:15:00 | 438.95 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2024-02-01 12:00:00 | 507.65 | 2024-02-02 09:15:00 | 533.03 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-01 12:00:00 | 507.65 | 2024-02-05 12:15:00 | 558.41 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-02-08 09:15:00 | 549.10 | 2024-02-09 09:15:00 | 537.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-02-19 09:15:00 | 535.10 | 2024-02-22 11:15:00 | 529.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-02-29 09:15:00 | 521.65 | 2024-03-01 14:15:00 | 527.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-03-01 09:45:00 | 525.75 | 2024-03-01 14:15:00 | 527.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-03-04 13:45:00 | 531.40 | 2024-03-06 09:15:00 | 522.65 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-03-05 11:00:00 | 532.00 | 2024-03-06 09:15:00 | 522.65 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-03-05 12:15:00 | 528.95 | 2024-03-06 09:15:00 | 522.65 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-03-05 12:45:00 | 530.85 | 2024-03-06 09:15:00 | 522.65 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-03-06 09:15:00 | 534.60 | 2024-03-06 09:15:00 | 522.65 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-03-18 09:15:00 | 503.05 | 2024-03-20 10:15:00 | 477.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 11:00:00 | 501.45 | 2024-03-20 10:15:00 | 476.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 14:30:00 | 500.55 | 2024-03-20 10:15:00 | 475.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 09:15:00 | 503.05 | 2024-03-21 09:15:00 | 479.00 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2024-03-18 11:00:00 | 501.45 | 2024-03-21 09:15:00 | 479.00 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2024-03-18 14:30:00 | 500.55 | 2024-03-21 09:15:00 | 479.00 | STOP_HIT | 0.50 | 4.31% |
| BUY | retest2 | 2024-03-28 09:15:00 | 508.00 | 2024-04-09 13:15:00 | 523.50 | STOP_HIT | 1.00 | 3.05% |
| BUY | retest2 | 2024-04-24 09:15:00 | 519.25 | 2024-04-24 11:15:00 | 515.75 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-04-24 10:30:00 | 517.70 | 2024-04-24 11:15:00 | 515.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-04-24 11:00:00 | 517.55 | 2024-04-24 11:15:00 | 515.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-04-25 11:45:00 | 514.85 | 2024-04-25 13:15:00 | 528.05 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-05-07 10:30:00 | 532.10 | 2024-05-13 09:15:00 | 505.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 13:30:00 | 532.70 | 2024-05-13 09:15:00 | 506.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 531.55 | 2024-05-13 09:15:00 | 504.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 530.45 | 2024-05-13 09:15:00 | 503.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 10:30:00 | 532.10 | 2024-05-13 13:15:00 | 513.15 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2024-05-08 13:30:00 | 532.70 | 2024-05-13 13:15:00 | 513.15 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2024-05-08 14:15:00 | 531.55 | 2024-05-13 13:15:00 | 513.15 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2024-05-09 09:15:00 | 530.45 | 2024-05-13 13:15:00 | 513.15 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2024-05-14 11:15:00 | 511.90 | 2024-05-15 09:15:00 | 527.25 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-05-21 11:15:00 | 543.00 | 2024-05-28 12:15:00 | 565.70 | STOP_HIT | 1.00 | 4.18% |
| SELL | retest2 | 2024-06-07 09:15:00 | 532.55 | 2024-06-11 14:15:00 | 538.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-07 11:30:00 | 539.45 | 2024-06-11 14:15:00 | 538.45 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-06-19 10:45:00 | 542.55 | 2024-06-24 14:15:00 | 541.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-06-19 11:15:00 | 543.75 | 2024-06-24 14:15:00 | 541.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-06-19 13:15:00 | 543.25 | 2024-06-24 15:15:00 | 541.20 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-20 10:15:00 | 545.10 | 2024-06-24 15:15:00 | 541.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-06-21 09:15:00 | 547.95 | 2024-06-24 15:15:00 | 541.20 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-06-21 12:45:00 | 547.00 | 2024-06-24 15:15:00 | 541.20 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-07-04 10:15:00 | 535.40 | 2024-07-04 14:15:00 | 543.55 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-07-04 12:45:00 | 536.00 | 2024-07-04 14:15:00 | 543.55 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-07-04 13:30:00 | 532.00 | 2024-07-04 14:15:00 | 543.55 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-07-05 13:15:00 | 535.85 | 2024-07-09 10:15:00 | 541.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-07-08 13:45:00 | 529.85 | 2024-07-09 10:15:00 | 541.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest1 | 2024-07-30 09:15:00 | 592.00 | 2024-08-01 13:15:00 | 595.95 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2024-08-27 14:45:00 | 548.20 | 2024-08-28 14:15:00 | 568.95 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-08-27 15:15:00 | 547.00 | 2024-08-28 14:15:00 | 568.95 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-08-28 14:15:00 | 547.65 | 2024-08-28 14:15:00 | 568.95 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-09-06 10:30:00 | 540.10 | 2024-09-09 09:15:00 | 513.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 10:30:00 | 540.10 | 2024-09-10 13:15:00 | 522.10 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2024-09-18 10:15:00 | 517.30 | 2024-09-23 09:15:00 | 522.55 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-09-26 14:45:00 | 532.45 | 2024-09-30 14:15:00 | 524.85 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-09-27 09:45:00 | 532.55 | 2024-09-30 14:15:00 | 524.85 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-09-30 11:00:00 | 532.00 | 2024-09-30 14:15:00 | 524.85 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-09-30 11:30:00 | 532.10 | 2024-09-30 14:15:00 | 524.85 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-01 14:15:00 | 525.75 | 2024-10-04 10:15:00 | 534.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-10-01 14:45:00 | 526.20 | 2024-10-04 10:15:00 | 534.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-01 15:15:00 | 526.20 | 2024-10-04 10:15:00 | 534.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-04 12:45:00 | 525.50 | 2024-10-08 13:15:00 | 527.35 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-10-07 10:45:00 | 512.90 | 2024-10-08 13:15:00 | 527.35 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-10-18 14:15:00 | 515.95 | 2024-10-25 09:15:00 | 490.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 516.50 | 2024-10-25 09:15:00 | 490.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 515.85 | 2024-10-25 09:15:00 | 490.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:15:00 | 515.95 | 2024-10-25 14:15:00 | 498.85 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2024-10-21 12:30:00 | 516.50 | 2024-10-25 14:15:00 | 498.85 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2024-10-21 14:00:00 | 515.85 | 2024-10-25 14:15:00 | 498.85 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2024-11-12 10:30:00 | 564.00 | 2024-11-14 09:15:00 | 536.18 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-11-12 10:30:00 | 564.00 | 2024-11-14 12:15:00 | 545.95 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-11-12 12:45:00 | 562.90 | 2024-11-14 15:15:00 | 535.80 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-11-12 13:45:00 | 564.40 | 2024-11-14 15:15:00 | 534.75 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2024-11-12 14:15:00 | 563.15 | 2024-11-14 15:15:00 | 534.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 562.90 | 2024-11-19 09:15:00 | 531.30 | STOP_HIT | 0.50 | 5.61% |
| SELL | retest2 | 2024-11-12 13:45:00 | 564.40 | 2024-11-19 09:15:00 | 531.30 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2024-11-12 14:15:00 | 563.15 | 2024-11-19 09:15:00 | 531.30 | STOP_HIT | 0.50 | 5.66% |
| BUY | retest2 | 2024-11-28 14:15:00 | 578.10 | 2024-12-11 10:15:00 | 592.15 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2024-11-29 09:15:00 | 582.00 | 2024-12-11 10:15:00 | 592.15 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2024-12-02 14:00:00 | 580.50 | 2024-12-11 10:15:00 | 592.15 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2024-12-03 14:30:00 | 577.15 | 2024-12-11 10:15:00 | 592.15 | STOP_HIT | 1.00 | 2.60% |
| SELL | retest2 | 2024-12-16 12:00:00 | 580.25 | 2024-12-16 14:15:00 | 582.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-16 13:00:00 | 582.55 | 2024-12-16 14:15:00 | 582.00 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-12-26 12:15:00 | 541.75 | 2024-12-26 14:15:00 | 545.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-06 09:15:00 | 513.00 | 2025-01-10 11:15:00 | 487.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 513.00 | 2025-01-14 09:15:00 | 485.35 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2025-01-23 11:30:00 | 519.30 | 2025-01-27 09:15:00 | 493.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:45:00 | 521.80 | 2025-01-27 09:15:00 | 495.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:45:00 | 521.20 | 2025-01-27 09:15:00 | 495.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 519.30 | 2025-01-27 12:15:00 | 500.50 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-01-23 12:45:00 | 521.80 | 2025-01-27 12:15:00 | 500.50 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-01-23 13:45:00 | 521.20 | 2025-01-27 12:15:00 | 500.50 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-02-05 15:15:00 | 542.00 | 2025-02-10 09:15:00 | 530.45 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-02-07 14:45:00 | 541.60 | 2025-02-10 09:15:00 | 530.45 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-02-12 14:30:00 | 515.55 | 2025-02-13 10:15:00 | 523.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-02-14 09:15:00 | 513.50 | 2025-02-18 13:15:00 | 515.15 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-02-14 10:00:00 | 514.00 | 2025-02-18 13:15:00 | 515.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-02-18 09:15:00 | 507.60 | 2025-02-18 13:15:00 | 515.15 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-03-04 11:30:00 | 503.30 | 2025-03-05 09:15:00 | 515.90 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-04 13:00:00 | 503.85 | 2025-03-05 09:15:00 | 515.90 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-04 13:30:00 | 500.10 | 2025-03-05 09:15:00 | 515.90 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-03-04 14:15:00 | 503.90 | 2025-03-05 09:15:00 | 515.90 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-03-26 09:15:00 | 548.55 | 2025-03-28 14:15:00 | 540.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-03-26 14:15:00 | 546.55 | 2025-03-28 14:15:00 | 540.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-03-27 09:30:00 | 546.85 | 2025-03-28 14:15:00 | 540.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-27 14:45:00 | 547.10 | 2025-03-28 14:15:00 | 540.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-04-25 15:15:00 | 566.95 | 2025-04-28 12:15:00 | 587.50 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-05-08 14:00:00 | 561.10 | 2025-05-12 10:15:00 | 570.35 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-20 13:15:00 | 612.00 | 2025-05-21 11:15:00 | 596.30 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-05-20 15:00:00 | 611.20 | 2025-05-21 11:15:00 | 596.30 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-05-26 12:15:00 | 595.40 | 2025-05-28 09:15:00 | 599.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-27 15:15:00 | 595.30 | 2025-05-28 09:15:00 | 599.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-05-29 15:00:00 | 601.45 | 2025-06-05 13:15:00 | 629.00 | STOP_HIT | 1.00 | 4.58% |
| BUY | retest2 | 2025-05-30 11:00:00 | 602.45 | 2025-06-05 13:15:00 | 629.00 | STOP_HIT | 1.00 | 4.41% |
| SELL | retest2 | 2025-06-20 15:15:00 | 615.00 | 2025-06-24 09:15:00 | 627.10 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-06-30 09:15:00 | 634.50 | 2025-07-07 15:15:00 | 643.85 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2025-07-10 09:30:00 | 636.75 | 2025-07-14 09:15:00 | 642.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-14 10:30:00 | 637.65 | 2025-07-14 14:15:00 | 635.10 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-07-21 14:45:00 | 631.45 | 2025-07-24 13:15:00 | 647.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-07-22 10:15:00 | 632.30 | 2025-07-24 13:15:00 | 647.10 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-07-25 13:00:00 | 635.35 | 2025-07-28 12:15:00 | 624.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-25 13:45:00 | 635.85 | 2025-07-28 12:15:00 | 624.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-07-25 14:30:00 | 635.30 | 2025-07-28 12:15:00 | 624.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-08-07 14:15:00 | 641.50 | 2025-08-20 11:15:00 | 670.50 | STOP_HIT | 1.00 | 4.52% |
| SELL | retest2 | 2025-08-29 15:15:00 | 651.00 | 2025-09-01 09:15:00 | 660.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 668.20 | 2025-09-05 11:15:00 | 661.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-04 15:15:00 | 665.40 | 2025-09-05 11:15:00 | 661.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-09 09:30:00 | 674.00 | 2025-09-23 12:15:00 | 698.10 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-09-09 10:30:00 | 674.00 | 2025-09-23 12:15:00 | 698.10 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-09-09 11:30:00 | 674.50 | 2025-09-23 12:15:00 | 698.10 | STOP_HIT | 1.00 | 3.50% |
| BUY | retest2 | 2025-09-10 09:15:00 | 677.80 | 2025-09-23 12:15:00 | 698.10 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2025-09-10 10:15:00 | 688.85 | 2025-09-23 12:15:00 | 698.10 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-10-01 13:30:00 | 739.15 | 2025-10-14 13:15:00 | 765.50 | STOP_HIT | 1.00 | 3.56% |
| BUY | retest2 | 2025-10-03 09:15:00 | 748.55 | 2025-10-14 13:15:00 | 765.50 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2025-10-16 09:15:00 | 778.70 | 2025-10-16 09:15:00 | 769.75 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-06 10:45:00 | 875.40 | 2025-11-06 11:15:00 | 864.45 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-19 12:30:00 | 886.60 | 2025-11-21 09:15:00 | 872.35 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-19 13:30:00 | 886.45 | 2025-11-21 09:15:00 | 872.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-11-19 15:00:00 | 886.50 | 2025-11-21 09:15:00 | 872.35 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-11-20 09:15:00 | 887.85 | 2025-11-21 09:15:00 | 872.35 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-05 12:15:00 | 809.90 | 2025-12-09 09:15:00 | 769.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:45:00 | 809.55 | 2025-12-09 09:15:00 | 769.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 809.80 | 2025-12-09 09:15:00 | 769.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 12:15:00 | 809.90 | 2025-12-09 10:15:00 | 789.35 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-12-05 13:45:00 | 809.55 | 2025-12-09 10:15:00 | 789.35 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-12-05 14:30:00 | 809.80 | 2025-12-09 10:15:00 | 789.35 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2026-01-09 12:15:00 | 843.80 | 2026-01-14 13:15:00 | 843.00 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2026-01-09 13:00:00 | 840.10 | 2026-01-14 13:15:00 | 843.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-02-11 11:30:00 | 901.25 | 2026-02-12 09:15:00 | 885.10 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-02-11 13:30:00 | 901.40 | 2026-02-12 09:15:00 | 885.10 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-02-20 09:15:00 | 933.20 | 2026-03-02 12:15:00 | 968.00 | STOP_HIT | 1.00 | 3.73% |
| SELL | retest2 | 2026-03-05 11:30:00 | 936.00 | 2026-03-09 09:15:00 | 889.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 12:30:00 | 943.00 | 2026-03-09 09:15:00 | 895.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 941.00 | 2026-03-09 09:15:00 | 893.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 943.10 | 2026-03-09 09:15:00 | 895.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:30:00 | 936.00 | 2026-03-10 09:15:00 | 909.95 | STOP_HIT | 0.50 | 2.78% |
| SELL | retest2 | 2026-03-05 12:30:00 | 943.00 | 2026-03-10 09:15:00 | 909.95 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2026-03-05 13:45:00 | 941.00 | 2026-03-10 09:15:00 | 909.95 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2026-03-06 14:30:00 | 943.10 | 2026-03-10 09:15:00 | 909.95 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2026-03-17 12:00:00 | 867.00 | 2026-03-18 10:15:00 | 886.65 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-04-01 10:15:00 | 864.80 | 2026-04-01 12:15:00 | 886.65 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-01 11:00:00 | 864.30 | 2026-04-01 12:15:00 | 886.65 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-04-13 10:45:00 | 950.00 | 2026-04-16 09:15:00 | 939.75 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-16 09:15:00 | 952.15 | 2026-04-16 09:15:00 | 939.75 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-04-20 13:45:00 | 932.20 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-04-20 14:30:00 | 931.75 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2026-04-21 10:15:00 | 932.00 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2026-04-21 11:15:00 | 930.80 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2026-04-22 12:30:00 | 926.70 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-04-22 15:00:00 | 925.85 | 2026-04-29 12:15:00 | 921.25 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2026-05-06 11:00:00 | 857.05 | 2026-05-06 14:15:00 | 865.70 | STOP_HIT | 1.00 | -1.01% |
