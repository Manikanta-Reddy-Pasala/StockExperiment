# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 750.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 205 |
| ALERT1 | 149 |
| ALERT2 | 146 |
| ALERT2_SKIP | 79 |
| ALERT3 | 392 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 169 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 164 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 182 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 127
- **Target hits / Stop hits / Partials:** 9 / 164 / 9
- **Avg / median % per leg:** 0.09% / -1.06%
- **Sum % (uncompounded):** 16.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 91 | 30 | 33.0% | 8 | 83 | 0 | 0.22% | 19.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 91 | 30 | 33.0% | 8 | 83 | 0 | 0.22% | 19.9% |
| SELL (all) | 91 | 25 | 27.5% | 1 | 81 | 9 | -0.04% | -3.8% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.05% | 4.2% |
| SELL @ 3rd Alert (retest2) | 87 | 22 | 25.3% | 1 | 77 | 9 | -0.09% | -8.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.05% | 4.2% |
| retest2 (combined) | 178 | 52 | 29.2% | 9 | 160 | 9 | 0.07% | 11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 15:15:00 | 279.50 | 278.77 | 278.69 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 276.75 | 278.36 | 278.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 11:15:00 | 274.00 | 277.25 | 277.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 11:15:00 | 275.30 | 274.89 | 276.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 12:00:00 | 275.30 | 274.89 | 276.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 275.55 | 275.05 | 275.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 13:30:00 | 276.15 | 275.05 | 275.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 276.40 | 275.32 | 275.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 276.40 | 275.32 | 275.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 277.30 | 275.71 | 276.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:15:00 | 278.30 | 275.71 | 276.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 10:15:00 | 279.00 | 276.80 | 276.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 11:15:00 | 280.40 | 277.52 | 276.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 15:15:00 | 281.50 | 281.74 | 280.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 09:15:00 | 281.15 | 281.74 | 280.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 282.40 | 281.87 | 280.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 12:45:00 | 283.95 | 282.32 | 281.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:15:00 | 283.80 | 282.52 | 281.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:15:00 | 286.00 | 282.23 | 281.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 09:15:00 | 281.30 | 284.53 | 284.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 281.30 | 284.53 | 284.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 280.85 | 283.80 | 284.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 282.45 | 282.38 | 283.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-06 14:45:00 | 282.25 | 282.38 | 283.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 285.00 | 282.90 | 283.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 285.00 | 282.90 | 283.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 284.50 | 283.22 | 283.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 12:00:00 | 283.50 | 283.28 | 283.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 13:45:00 | 283.55 | 283.52 | 283.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 15:15:00 | 284.25 | 283.73 | 283.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 15:15:00 | 284.25 | 283.73 | 283.71 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 09:15:00 | 282.70 | 283.52 | 283.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 11:15:00 | 280.50 | 282.79 | 283.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 279.85 | 279.01 | 280.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 279.85 | 279.01 | 280.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 279.85 | 279.01 | 280.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:00:00 | 279.85 | 279.01 | 280.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 279.85 | 279.24 | 280.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:00:00 | 279.85 | 279.24 | 280.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 280.20 | 279.43 | 280.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:45:00 | 280.35 | 279.43 | 280.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 279.90 | 279.52 | 280.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:00:00 | 279.90 | 279.52 | 280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 279.25 | 279.47 | 280.12 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 284.90 | 281.03 | 280.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 12:15:00 | 285.45 | 282.44 | 281.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 285.50 | 286.56 | 285.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 285.50 | 286.56 | 285.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 285.50 | 286.56 | 285.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:30:00 | 285.25 | 286.56 | 285.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 287.30 | 286.99 | 285.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:45:00 | 285.60 | 286.99 | 285.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 293.60 | 294.15 | 292.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 293.65 | 294.15 | 292.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 293.15 | 293.50 | 292.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 14:30:00 | 292.65 | 293.50 | 292.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 292.20 | 293.24 | 292.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:15:00 | 291.60 | 293.24 | 292.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 287.90 | 292.17 | 292.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 282.60 | 288.22 | 289.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 14:15:00 | 287.55 | 287.35 | 288.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 14:15:00 | 287.55 | 287.35 | 288.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 287.55 | 287.35 | 288.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:30:00 | 289.00 | 287.35 | 288.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 289.05 | 287.46 | 288.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 289.05 | 287.46 | 288.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 289.85 | 287.93 | 288.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 12:00:00 | 289.85 | 287.93 | 288.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 289.25 | 288.38 | 288.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 289.25 | 288.38 | 288.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 291.35 | 288.97 | 288.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 15:15:00 | 291.70 | 289.52 | 289.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 09:15:00 | 295.20 | 298.68 | 296.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 295.20 | 298.68 | 296.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 295.20 | 298.68 | 296.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:45:00 | 294.95 | 298.68 | 296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 294.90 | 297.93 | 296.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:30:00 | 295.25 | 297.93 | 296.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 296.35 | 296.86 | 296.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 13:30:00 | 295.65 | 296.86 | 296.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 296.85 | 296.86 | 296.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 14:45:00 | 296.80 | 296.86 | 296.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 299.15 | 299.04 | 297.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 15:00:00 | 299.15 | 299.04 | 297.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 295.30 | 298.25 | 297.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 295.30 | 298.25 | 297.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 296.40 | 297.88 | 297.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:15:00 | 295.15 | 297.88 | 297.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 295.85 | 297.47 | 297.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 294.80 | 296.94 | 297.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 09:15:00 | 298.95 | 296.16 | 296.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 09:15:00 | 298.95 | 296.16 | 296.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 298.95 | 296.16 | 296.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:00:00 | 298.95 | 296.16 | 296.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 298.50 | 296.63 | 296.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 13:15:00 | 302.70 | 298.33 | 297.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 15:15:00 | 310.20 | 310.58 | 307.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:15:00 | 309.40 | 310.58 | 307.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 308.40 | 310.14 | 307.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 307.80 | 310.14 | 307.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 310.35 | 310.47 | 309.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 11:30:00 | 309.50 | 310.47 | 309.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 309.75 | 310.37 | 309.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:00:00 | 309.75 | 310.37 | 309.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 308.80 | 310.06 | 309.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:30:00 | 309.15 | 310.06 | 309.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 309.25 | 309.90 | 309.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 310.25 | 309.90 | 309.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 10:00:00 | 310.65 | 310.05 | 309.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:00:00 | 310.30 | 310.10 | 309.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 12:15:00 | 305.90 | 309.09 | 309.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 305.90 | 309.09 | 309.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 301.80 | 307.63 | 308.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 11:15:00 | 307.00 | 305.44 | 306.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 11:15:00 | 307.00 | 305.44 | 306.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 307.00 | 305.44 | 306.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 306.50 | 305.44 | 306.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 308.30 | 306.01 | 306.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:30:00 | 309.00 | 306.01 | 306.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 309.65 | 306.74 | 307.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:00:00 | 309.65 | 306.74 | 307.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 313.00 | 308.57 | 307.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 316.55 | 311.18 | 310.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 15:15:00 | 313.00 | 313.12 | 311.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 09:15:00 | 313.00 | 313.12 | 311.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 312.30 | 312.96 | 311.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:45:00 | 312.55 | 312.96 | 311.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 313.15 | 313.00 | 312.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 13:45:00 | 314.65 | 313.12 | 312.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 14:15:00 | 311.80 | 312.85 | 312.23 | SL hit (close<static) qty=1.00 sl=312.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 13:15:00 | 310.80 | 311.79 | 311.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 309.90 | 311.41 | 311.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 310.80 | 308.69 | 309.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 310.80 | 308.69 | 309.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 310.80 | 308.69 | 309.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 310.80 | 308.69 | 309.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 307.80 | 308.51 | 309.53 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 310.80 | 309.66 | 309.66 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 13:15:00 | 309.10 | 309.64 | 309.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 14:15:00 | 308.10 | 309.33 | 309.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 15:15:00 | 309.50 | 309.37 | 309.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:15:00 | 314.45 | 309.37 | 309.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 17 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 318.95 | 311.28 | 310.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 12:15:00 | 319.30 | 314.51 | 312.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 324.70 | 324.88 | 323.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 324.70 | 324.88 | 323.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 324.70 | 324.88 | 323.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 324.00 | 324.88 | 323.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 318.90 | 323.69 | 322.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:45:00 | 318.50 | 323.69 | 322.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 318.50 | 322.65 | 322.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 318.80 | 322.65 | 322.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 316.80 | 321.48 | 321.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 314.70 | 320.12 | 321.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 319.90 | 319.14 | 320.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 319.90 | 319.14 | 320.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 318.90 | 319.09 | 320.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 12:15:00 | 318.15 | 319.09 | 320.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 318.10 | 318.88 | 319.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 11:15:00 | 321.60 | 319.87 | 319.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 11:15:00 | 321.60 | 319.87 | 319.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 324.05 | 321.09 | 320.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 320.90 | 321.05 | 320.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 320.90 | 321.05 | 320.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 320.90 | 321.05 | 320.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 320.90 | 321.05 | 320.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 323.80 | 321.60 | 320.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 13:15:00 | 324.40 | 321.94 | 320.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 324.70 | 322.72 | 321.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 11:30:00 | 324.00 | 322.92 | 322.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 09:15:00 | 310.60 | 320.16 | 321.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 09:15:00 | 310.60 | 320.16 | 321.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 304.25 | 312.62 | 316.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 293.70 | 293.20 | 298.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 09:45:00 | 294.00 | 293.20 | 298.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 294.70 | 293.77 | 295.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:45:00 | 296.40 | 293.77 | 295.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 295.30 | 294.08 | 295.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 15:00:00 | 295.30 | 294.08 | 295.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 292.95 | 293.85 | 295.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:15:00 | 295.10 | 293.85 | 295.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 291.75 | 293.43 | 294.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 12:45:00 | 289.95 | 292.34 | 294.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:15:00 | 290.00 | 291.71 | 293.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:00:00 | 290.35 | 291.17 | 292.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 14:15:00 | 298.45 | 294.38 | 293.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 298.45 | 294.38 | 293.90 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 14:15:00 | 294.25 | 295.87 | 295.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 293.50 | 295.39 | 295.65 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 298.85 | 296.08 | 295.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 12:15:00 | 300.45 | 297.91 | 296.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 15:15:00 | 298.15 | 298.62 | 297.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 15:15:00 | 298.15 | 298.62 | 297.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 298.15 | 298.62 | 297.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:15:00 | 301.00 | 298.62 | 297.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 14:15:00 | 297.65 | 300.36 | 300.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 297.65 | 300.36 | 300.61 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 301.80 | 300.79 | 300.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 303.00 | 301.65 | 301.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 12:15:00 | 310.60 | 310.62 | 308.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 12:45:00 | 310.50 | 310.62 | 308.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 313.25 | 315.20 | 313.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 15:00:00 | 313.25 | 315.20 | 313.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 312.70 | 314.70 | 313.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 316.45 | 314.70 | 313.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 310.85 | 313.84 | 313.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 310.85 | 313.84 | 313.95 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 10:15:00 | 318.50 | 313.65 | 313.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 323.90 | 319.90 | 317.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 331.15 | 331.27 | 326.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 14:45:00 | 331.40 | 331.27 | 326.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 327.00 | 329.31 | 328.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:00:00 | 327.00 | 329.31 | 328.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 326.60 | 328.77 | 327.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 324.25 | 328.77 | 327.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 333.00 | 329.44 | 328.37 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 325.25 | 328.45 | 328.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 323.50 | 327.46 | 328.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 327.35 | 326.08 | 327.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 11:15:00 | 327.35 | 326.08 | 327.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 327.35 | 326.08 | 327.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 11:45:00 | 327.70 | 326.08 | 327.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 333.20 | 327.50 | 327.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 13:00:00 | 333.20 | 327.50 | 327.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 13:15:00 | 335.05 | 329.01 | 328.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 11:15:00 | 337.55 | 333.86 | 331.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 15:15:00 | 330.10 | 333.95 | 332.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 15:15:00 | 330.10 | 333.95 | 332.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 330.10 | 333.95 | 332.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 332.25 | 333.72 | 332.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 334.80 | 333.94 | 332.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 332.80 | 333.94 | 332.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 342.70 | 337.14 | 334.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 10:45:00 | 345.05 | 338.83 | 335.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 10:15:00 | 345.70 | 345.79 | 341.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 12:15:00 | 347.30 | 345.52 | 342.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 13:45:00 | 345.70 | 345.20 | 342.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 340.40 | 344.24 | 342.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 340.40 | 344.24 | 342.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 344.25 | 344.24 | 342.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 344.70 | 344.24 | 342.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 15:15:00 | 354.25 | 357.93 | 357.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 15:15:00 | 354.25 | 357.93 | 357.99 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 360.75 | 358.49 | 358.24 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 356.15 | 358.06 | 358.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 10:15:00 | 354.40 | 357.33 | 357.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 355.00 | 354.69 | 356.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 355.00 | 354.69 | 356.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 355.00 | 354.69 | 356.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 15:15:00 | 352.40 | 355.10 | 355.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 11:00:00 | 353.10 | 354.47 | 355.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 14:15:00 | 352.80 | 354.04 | 354.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 10:00:00 | 352.15 | 353.16 | 354.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 353.15 | 353.16 | 354.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:00:00 | 353.15 | 353.16 | 354.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 353.00 | 353.13 | 353.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:30:00 | 353.85 | 353.13 | 353.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 354.55 | 353.41 | 353.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:00:00 | 354.55 | 353.41 | 353.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 354.85 | 353.70 | 354.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:45:00 | 355.35 | 353.70 | 354.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 355.40 | 354.04 | 354.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 15:15:00 | 353.45 | 354.04 | 354.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 356.95 | 354.53 | 354.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 356.95 | 354.53 | 354.38 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 352.70 | 353.99 | 354.15 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 13:15:00 | 356.50 | 354.61 | 354.41 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 352.25 | 353.98 | 354.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 10:15:00 | 349.00 | 352.01 | 353.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 327.15 | 325.48 | 329.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 327.15 | 325.48 | 329.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 327.15 | 325.48 | 329.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 328.70 | 325.48 | 329.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 323.35 | 325.48 | 327.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 10:15:00 | 321.95 | 324.79 | 326.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 09:15:00 | 332.80 | 326.72 | 326.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 332.80 | 326.72 | 326.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 10:15:00 | 334.95 | 328.36 | 327.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 10:15:00 | 347.60 | 347.84 | 343.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 10:30:00 | 347.80 | 347.84 | 343.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 358.60 | 357.52 | 355.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:00:00 | 358.60 | 357.52 | 355.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 370.00 | 367.22 | 363.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 10:45:00 | 370.45 | 367.58 | 364.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:15:00 | 370.60 | 368.59 | 366.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 372.20 | 366.95 | 366.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 11:00:00 | 374.15 | 368.71 | 367.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 371.60 | 371.53 | 369.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 371.60 | 371.53 | 369.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 367.05 | 370.42 | 369.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:00:00 | 367.05 | 370.42 | 369.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 366.15 | 369.57 | 369.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:30:00 | 365.00 | 369.57 | 369.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-20 13:15:00 | 365.20 | 368.69 | 368.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 365.20 | 368.69 | 368.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 362.80 | 366.81 | 367.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 11:15:00 | 366.70 | 364.62 | 365.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 11:15:00 | 366.70 | 364.62 | 365.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 366.70 | 364.62 | 365.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 11:45:00 | 366.05 | 364.62 | 365.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 366.70 | 365.03 | 365.78 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 14:15:00 | 371.20 | 366.73 | 366.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 09:15:00 | 371.55 | 368.15 | 367.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 09:15:00 | 380.60 | 381.32 | 375.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 09:30:00 | 378.80 | 381.32 | 375.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 378.50 | 380.33 | 377.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 13:45:00 | 377.55 | 380.33 | 377.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 378.00 | 379.49 | 377.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 382.85 | 379.49 | 377.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 09:15:00 | 389.60 | 392.64 | 392.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 389.60 | 392.64 | 392.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 393.15 | 391.42 | 391.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 12:15:00 | 394.95 | 392.33 | 391.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 390.85 | 393.21 | 392.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 10:15:00 | 390.85 | 393.21 | 392.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 390.85 | 393.21 | 392.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 390.85 | 393.21 | 392.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 389.70 | 392.51 | 392.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 12:00:00 | 389.70 | 392.51 | 392.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 387.15 | 391.44 | 391.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 11:15:00 | 385.10 | 387.73 | 389.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 389.45 | 386.86 | 388.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 389.45 | 386.86 | 388.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 389.45 | 386.86 | 388.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:45:00 | 391.85 | 386.86 | 388.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 391.50 | 387.79 | 388.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:00:00 | 391.50 | 387.79 | 388.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 390.50 | 388.33 | 388.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:45:00 | 389.15 | 388.51 | 388.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 14:15:00 | 387.50 | 388.75 | 388.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 09:15:00 | 391.45 | 387.41 | 387.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 391.45 | 387.41 | 387.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 12:15:00 | 397.20 | 392.94 | 390.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 400.10 | 400.54 | 397.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 15:00:00 | 400.10 | 400.54 | 397.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 397.35 | 400.87 | 398.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 397.35 | 400.87 | 398.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 386.75 | 398.04 | 397.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 391.35 | 398.04 | 397.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 376.75 | 393.78 | 395.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 376.50 | 390.33 | 393.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 388.45 | 385.07 | 388.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 388.45 | 385.07 | 388.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 388.45 | 385.07 | 388.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:30:00 | 387.35 | 385.07 | 388.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 392.55 | 386.56 | 388.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 392.55 | 386.56 | 388.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 391.95 | 387.64 | 389.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 388.15 | 387.26 | 388.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:00:00 | 385.75 | 387.26 | 388.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 13:15:00 | 391.15 | 389.02 | 388.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 13:15:00 | 391.15 | 389.02 | 388.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 396.75 | 390.57 | 389.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 395.60 | 395.82 | 393.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 395.60 | 395.82 | 393.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 395.60 | 395.82 | 393.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:00:00 | 395.60 | 395.82 | 393.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 398.70 | 396.40 | 394.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 12:45:00 | 401.50 | 397.86 | 396.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 13:15:00 | 412.80 | 416.55 | 416.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 412.80 | 416.55 | 416.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 15:15:00 | 411.10 | 414.99 | 415.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 406.50 | 405.37 | 409.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 10:00:00 | 406.50 | 405.37 | 409.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 409.85 | 406.23 | 408.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:00:00 | 409.85 | 406.23 | 408.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 410.00 | 406.98 | 409.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:30:00 | 410.40 | 406.98 | 409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 410.00 | 408.46 | 409.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 406.90 | 408.46 | 409.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 405.25 | 407.82 | 408.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:00:00 | 404.10 | 406.94 | 408.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:45:00 | 402.00 | 406.23 | 407.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 15:15:00 | 411.35 | 408.06 | 407.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 15:15:00 | 411.35 | 408.06 | 407.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 413.20 | 409.09 | 408.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 14:15:00 | 409.70 | 409.76 | 408.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-12 15:00:00 | 409.70 | 409.76 | 408.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 409.15 | 409.64 | 408.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 411.75 | 409.64 | 408.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 412.00 | 410.11 | 409.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 10:15:00 | 419.90 | 410.11 | 409.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 416.10 | 421.62 | 421.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 416.10 | 421.62 | 421.65 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 424.95 | 421.68 | 421.42 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 418.70 | 421.44 | 421.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 417.00 | 419.89 | 420.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 12:15:00 | 424.25 | 419.19 | 419.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 12:15:00 | 424.25 | 419.19 | 419.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 424.25 | 419.19 | 419.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:45:00 | 424.15 | 419.19 | 419.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 417.55 | 418.87 | 419.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 14:15:00 | 413.75 | 418.87 | 419.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 10:00:00 | 415.15 | 416.51 | 418.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 11:00:00 | 415.40 | 416.29 | 418.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:45:00 | 414.05 | 416.02 | 416.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 414.60 | 415.74 | 416.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:45:00 | 414.00 | 415.74 | 416.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 413.00 | 414.02 | 415.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 413.85 | 414.02 | 415.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 411.00 | 413.41 | 414.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 10:30:00 | 409.00 | 412.98 | 414.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 13:15:00 | 410.05 | 412.63 | 414.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 10:30:00 | 410.40 | 411.12 | 412.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 15:00:00 | 409.20 | 411.86 | 412.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 414.15 | 411.85 | 412.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 414.15 | 411.85 | 412.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 414.35 | 412.35 | 412.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:30:00 | 414.00 | 412.35 | 412.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-31 12:15:00 | 416.30 | 413.38 | 413.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 12:15:00 | 416.30 | 413.38 | 413.07 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 409.90 | 412.80 | 413.02 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 421.55 | 414.42 | 413.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 426.45 | 419.93 | 416.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 12:15:00 | 426.20 | 427.42 | 424.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-06 13:15:00 | 425.45 | 427.42 | 424.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 426.80 | 427.30 | 424.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 13:30:00 | 424.65 | 427.30 | 424.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 430.20 | 432.78 | 430.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 430.20 | 432.78 | 430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 431.80 | 432.58 | 430.21 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 419.15 | 427.98 | 428.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 414.70 | 425.32 | 427.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 424.65 | 421.52 | 424.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 424.65 | 421.52 | 424.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 424.65 | 421.52 | 424.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:30:00 | 428.75 | 421.52 | 424.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 421.25 | 421.46 | 423.91 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 12:15:00 | 426.50 | 424.06 | 423.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 13:15:00 | 428.20 | 424.89 | 424.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 09:15:00 | 425.35 | 426.20 | 425.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 425.35 | 426.20 | 425.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 425.35 | 426.20 | 425.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 10:45:00 | 428.05 | 426.44 | 425.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 13:30:00 | 428.90 | 426.98 | 425.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-19 09:15:00 | 470.86 | 455.34 | 445.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 462.35 | 466.64 | 467.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 459.75 | 465.26 | 466.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 464.45 | 461.00 | 462.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 464.45 | 461.00 | 462.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 464.45 | 461.00 | 462.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 464.45 | 461.00 | 462.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 466.95 | 462.19 | 463.11 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 465.30 | 463.84 | 463.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 465.55 | 464.18 | 463.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 14:15:00 | 462.35 | 463.81 | 463.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 14:15:00 | 462.35 | 463.81 | 463.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 462.35 | 463.81 | 463.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 462.35 | 463.81 | 463.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 465.50 | 464.15 | 463.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-02 09:15:00 | 467.75 | 464.15 | 463.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 465.55 | 464.59 | 464.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:45:00 | 466.00 | 467.24 | 466.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 13:00:00 | 465.80 | 467.17 | 466.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 467.25 | 467.37 | 466.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:45:00 | 466.70 | 467.37 | 466.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 456.35 | 465.15 | 465.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 456.35 | 465.15 | 465.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 446.90 | 459.72 | 463.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 458.70 | 457.90 | 461.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 15:00:00 | 458.70 | 457.90 | 461.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 459.60 | 458.40 | 460.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:30:00 | 454.85 | 456.98 | 459.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 432.11 | 442.48 | 449.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 14:15:00 | 409.37 | 420.28 | 430.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 420.80 | 415.76 | 415.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 11:15:00 | 422.45 | 417.10 | 416.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 431.40 | 432.90 | 429.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 09:45:00 | 432.75 | 432.90 | 429.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 432.00 | 432.49 | 430.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:30:00 | 430.85 | 432.49 | 430.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 430.40 | 432.07 | 430.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 430.40 | 432.07 | 430.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 430.00 | 431.66 | 430.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 428.90 | 431.66 | 430.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 428.65 | 431.06 | 430.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:00:00 | 430.70 | 430.98 | 430.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 15:15:00 | 431.75 | 431.16 | 430.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 12:15:00 | 442.25 | 446.03 | 446.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 12:15:00 | 442.25 | 446.03 | 446.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 14:15:00 | 438.95 | 443.92 | 445.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 11:15:00 | 431.00 | 429.09 | 434.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-10 12:00:00 | 431.00 | 429.09 | 434.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 433.10 | 429.08 | 432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 434.25 | 429.08 | 432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 430.65 | 429.39 | 432.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 11:30:00 | 430.00 | 429.44 | 431.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 13:15:00 | 408.50 | 416.87 | 422.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 10:15:00 | 416.65 | 414.71 | 419.76 | SL hit (close>ema200) qty=0.50 sl=414.71 alert=retest2 |

### Cycle 61 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 418.30 | 415.21 | 415.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 10:15:00 | 420.45 | 417.09 | 416.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 417.80 | 418.17 | 417.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 15:00:00 | 417.80 | 418.17 | 417.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 417.50 | 418.04 | 417.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 417.70 | 418.04 | 417.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 415.70 | 417.57 | 417.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 415.70 | 417.57 | 417.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 414.90 | 417.04 | 416.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 413.40 | 417.04 | 416.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 11:15:00 | 414.20 | 416.47 | 416.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 12:15:00 | 413.35 | 415.84 | 416.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 418.40 | 416.32 | 416.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 14:15:00 | 418.40 | 416.32 | 416.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 418.40 | 416.32 | 416.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:00:00 | 418.40 | 416.32 | 416.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 416.10 | 416.28 | 416.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 417.25 | 416.28 | 416.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 416.65 | 416.35 | 416.42 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 417.40 | 416.56 | 416.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 11:15:00 | 422.80 | 417.81 | 417.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 419.60 | 420.93 | 419.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 419.60 | 420.93 | 419.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 421.75 | 421.09 | 419.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:00:00 | 422.05 | 421.28 | 419.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 422.75 | 421.01 | 420.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 10:00:00 | 425.80 | 421.96 | 420.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 10:45:00 | 423.00 | 423.04 | 422.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 420.90 | 422.61 | 422.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:00:00 | 420.90 | 422.61 | 422.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 424.30 | 422.95 | 422.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 14:30:00 | 425.75 | 423.51 | 422.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 425.80 | 423.51 | 422.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:30:00 | 425.25 | 424.42 | 423.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 12:15:00 | 425.10 | 424.28 | 423.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 424.65 | 424.36 | 423.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 418.55 | 423.24 | 423.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 418.55 | 423.24 | 423.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 414.30 | 420.35 | 421.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 12:15:00 | 412.05 | 410.21 | 414.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 13:00:00 | 412.05 | 410.21 | 414.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 397.70 | 397.55 | 401.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 395.00 | 397.55 | 401.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 400.60 | 398.65 | 400.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:00:00 | 400.60 | 398.65 | 400.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 402.15 | 399.35 | 400.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 402.15 | 399.35 | 400.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 405.40 | 400.56 | 401.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 405.40 | 400.56 | 401.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 407.20 | 401.89 | 401.84 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 399.50 | 401.41 | 401.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 11:15:00 | 398.90 | 400.79 | 401.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 14:15:00 | 400.05 | 400.01 | 400.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 15:00:00 | 400.05 | 400.01 | 400.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 398.50 | 399.71 | 400.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 398.45 | 399.71 | 400.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 397.95 | 399.36 | 400.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 10:15:00 | 396.40 | 399.36 | 400.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 12:15:00 | 404.95 | 399.11 | 399.86 | SL hit (close>static) qty=1.00 sl=403.35 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 402.40 | 398.73 | 398.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 417.90 | 408.96 | 405.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 426.45 | 427.45 | 421.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:30:00 | 426.00 | 427.45 | 421.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 438.30 | 430.80 | 426.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 444.50 | 436.79 | 431.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 430.85 | 432.65 | 432.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 430.85 | 432.65 | 432.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 429.00 | 431.24 | 431.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 422.75 | 422.60 | 425.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:00:00 | 422.75 | 422.60 | 425.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 428.70 | 423.00 | 425.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 428.70 | 423.00 | 425.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 430.05 | 424.41 | 425.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:00:00 | 430.05 | 424.41 | 425.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 437.80 | 427.09 | 426.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 441.50 | 429.97 | 427.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 431.05 | 434.67 | 431.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 431.05 | 434.67 | 431.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 431.05 | 434.67 | 431.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 423.95 | 434.67 | 431.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 411.00 | 429.94 | 429.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 411.00 | 429.94 | 429.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 402.45 | 424.44 | 426.99 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 435.65 | 426.68 | 425.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 438.45 | 429.04 | 427.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 473.60 | 473.84 | 464.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 473.60 | 473.84 | 464.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 472.65 | 476.02 | 472.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 472.00 | 476.02 | 472.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 472.60 | 474.27 | 472.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 472.60 | 474.27 | 472.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 472.15 | 473.84 | 472.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:30:00 | 471.30 | 473.84 | 472.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 471.90 | 473.45 | 472.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 470.60 | 473.45 | 472.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 10:15:00 | 469.45 | 471.91 | 471.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 466.85 | 470.28 | 471.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 465.05 | 462.20 | 464.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 465.05 | 462.20 | 464.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 465.05 | 462.20 | 464.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 468.00 | 462.20 | 464.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 466.85 | 463.13 | 465.04 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 474.00 | 467.61 | 466.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 493.90 | 476.09 | 472.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 487.90 | 488.52 | 482.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 487.90 | 488.52 | 482.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 487.90 | 488.52 | 482.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 484.25 | 488.52 | 482.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 490.20 | 489.87 | 485.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 497.15 | 489.87 | 485.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 492.40 | 491.21 | 488.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:00:00 | 491.45 | 491.26 | 489.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 492.00 | 492.71 | 490.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 501.40 | 498.74 | 495.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 10:45:00 | 503.60 | 499.78 | 496.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 491.75 | 497.02 | 496.25 | SL hit (close<static) qty=1.00 sl=492.90 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 491.80 | 495.41 | 495.62 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 499.60 | 495.71 | 495.52 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 489.50 | 494.71 | 495.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 487.70 | 493.31 | 494.44 | Break + close below crossover candle low |

### Cycle 77 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 503.90 | 494.52 | 494.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 11:15:00 | 508.90 | 498.69 | 496.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 499.30 | 500.30 | 497.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:30:00 | 500.50 | 500.30 | 497.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 507.85 | 501.78 | 498.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 502.80 | 501.78 | 498.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 521.45 | 520.95 | 517.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 515.35 | 520.95 | 517.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 516.95 | 520.15 | 517.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:30:00 | 516.00 | 520.15 | 517.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 518.00 | 519.72 | 517.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 522.00 | 519.46 | 517.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 512.75 | 518.98 | 518.39 | SL hit (close<static) qty=1.00 sl=513.25 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 510.20 | 517.22 | 517.65 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 524.20 | 516.28 | 516.01 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 511.65 | 518.04 | 518.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 509.60 | 513.78 | 515.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 514.40 | 510.75 | 513.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 514.40 | 510.75 | 513.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 514.40 | 510.75 | 513.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 514.40 | 510.75 | 513.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 515.05 | 511.61 | 513.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 516.65 | 511.61 | 513.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 516.65 | 512.62 | 514.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 516.65 | 512.62 | 514.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 515.65 | 513.22 | 514.19 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 517.00 | 514.98 | 514.84 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 509.25 | 513.84 | 514.33 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 523.65 | 515.17 | 514.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 526.75 | 517.49 | 515.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 560.40 | 562.68 | 557.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 09:45:00 | 560.60 | 562.68 | 557.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 571.45 | 564.44 | 558.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:15:00 | 577.55 | 565.95 | 559.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-02 09:15:00 | 635.30 | 630.87 | 616.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 667.25 | 670.81 | 670.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 14:15:00 | 664.55 | 668.28 | 669.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 11:15:00 | 665.95 | 665.06 | 667.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 11:15:00 | 665.95 | 665.06 | 667.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 665.95 | 665.06 | 667.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:45:00 | 667.65 | 665.06 | 667.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 676.50 | 666.54 | 666.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 676.55 | 666.54 | 666.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 671.70 | 667.57 | 667.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 684.05 | 674.97 | 671.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 684.95 | 686.37 | 681.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 13:00:00 | 684.95 | 686.37 | 681.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 681.70 | 684.96 | 682.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 689.05 | 684.26 | 683.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 679.10 | 683.32 | 683.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 679.10 | 683.32 | 683.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 673.35 | 681.19 | 682.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 688.60 | 679.12 | 680.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 688.60 | 679.12 | 680.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 688.60 | 679.12 | 680.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 688.60 | 679.12 | 680.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 688.30 | 680.95 | 681.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:15:00 | 690.25 | 680.95 | 681.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 692.00 | 683.16 | 682.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 698.35 | 687.32 | 684.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 707.60 | 710.75 | 704.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 707.60 | 710.75 | 704.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 698.00 | 708.20 | 704.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 698.00 | 708.20 | 704.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 690.50 | 704.66 | 702.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:15:00 | 688.35 | 704.66 | 702.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 687.45 | 701.22 | 701.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 682.90 | 697.55 | 699.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 693.60 | 689.42 | 692.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 693.60 | 689.42 | 692.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 693.60 | 689.42 | 692.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:30:00 | 696.00 | 689.42 | 692.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 695.65 | 690.66 | 692.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:00:00 | 695.65 | 690.66 | 692.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 699.10 | 692.35 | 693.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:30:00 | 698.60 | 692.35 | 693.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 703.35 | 694.55 | 694.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 705.00 | 696.64 | 695.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 11:15:00 | 701.75 | 702.94 | 699.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 11:15:00 | 701.75 | 702.94 | 699.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 701.75 | 702.94 | 699.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:00:00 | 701.75 | 702.94 | 699.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 705.50 | 703.45 | 699.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 706.75 | 701.29 | 699.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 11:15:00 | 699.10 | 700.83 | 699.98 | SL hit (close<static) qty=1.00 sl=699.40 alert=retest2 |

### Cycle 90 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 692.15 | 698.78 | 699.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 688.25 | 695.07 | 697.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 685.95 | 676.50 | 683.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 685.95 | 676.50 | 683.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 685.95 | 676.50 | 683.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 685.95 | 676.50 | 683.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 680.30 | 677.26 | 683.20 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 695.10 | 686.01 | 685.86 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 676.20 | 686.12 | 686.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 675.00 | 683.90 | 685.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 547.05 | 544.11 | 553.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 15:00:00 | 547.05 | 544.11 | 553.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 546.80 | 545.37 | 552.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 552.15 | 545.37 | 552.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 542.30 | 543.08 | 547.74 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 560.55 | 547.15 | 546.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 564.35 | 555.57 | 551.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 551.15 | 560.15 | 557.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 551.15 | 560.15 | 557.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 551.15 | 560.15 | 557.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 551.15 | 560.15 | 557.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 550.85 | 558.29 | 556.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:45:00 | 553.60 | 557.53 | 556.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 543.30 | 553.70 | 555.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 543.30 | 553.70 | 555.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 14:15:00 | 542.50 | 550.63 | 553.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 555.65 | 551.14 | 553.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 555.65 | 551.14 | 553.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 555.65 | 551.14 | 553.11 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 558.00 | 554.62 | 554.27 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 551.50 | 553.99 | 554.01 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 561.10 | 554.64 | 554.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 570.45 | 559.31 | 556.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 15:15:00 | 590.50 | 590.92 | 580.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:15:00 | 585.25 | 590.92 | 580.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 566.50 | 584.23 | 580.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 566.50 | 584.23 | 580.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 564.80 | 580.34 | 579.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 564.40 | 580.34 | 579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 569.70 | 578.21 | 578.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 552.30 | 571.73 | 575.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 556.95 | 555.47 | 562.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 554.45 | 555.47 | 562.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 560.00 | 556.80 | 562.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:30:00 | 562.10 | 556.80 | 562.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 559.35 | 557.67 | 560.80 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 571.25 | 562.72 | 562.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 573.95 | 564.96 | 563.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 564.00 | 567.21 | 565.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 564.00 | 567.21 | 565.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 564.00 | 567.21 | 565.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 564.00 | 567.21 | 565.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 574.20 | 568.61 | 565.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:45:00 | 567.15 | 568.61 | 565.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 596.30 | 599.84 | 595.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 596.30 | 599.84 | 595.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 600.35 | 599.95 | 596.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 608.00 | 599.98 | 596.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:30:00 | 604.50 | 600.30 | 597.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 603.00 | 600.54 | 597.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:45:00 | 602.10 | 603.64 | 600.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 600.00 | 602.91 | 600.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 595.35 | 602.91 | 600.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 599.15 | 602.16 | 600.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 599.00 | 602.16 | 600.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 594.00 | 600.53 | 600.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 594.00 | 600.53 | 600.02 | SL hit (close<static) qty=1.00 sl=595.95 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 593.05 | 599.03 | 599.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 589.55 | 596.41 | 598.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 597.40 | 595.46 | 597.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 597.40 | 595.46 | 597.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 597.40 | 595.46 | 597.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:00:00 | 589.50 | 595.64 | 596.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 587.90 | 593.22 | 595.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 560.02 | 576.09 | 585.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 558.50 | 576.09 | 585.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 574.35 | 573.09 | 581.52 | SL hit (close>ema200) qty=0.50 sl=573.09 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 557.55 | 552.30 | 552.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 571.70 | 560.07 | 556.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 557.35 | 563.64 | 560.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 557.35 | 563.64 | 560.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 557.35 | 563.64 | 560.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 557.35 | 563.64 | 560.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 550.65 | 561.04 | 559.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 550.65 | 561.04 | 559.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 549.80 | 558.79 | 558.30 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 552.25 | 557.48 | 557.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 544.40 | 551.91 | 554.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 548.80 | 547.37 | 551.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 548.80 | 547.37 | 551.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 551.50 | 548.20 | 551.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 551.50 | 548.20 | 551.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 548.60 | 548.28 | 551.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 559.35 | 548.28 | 551.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 559.85 | 550.59 | 551.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 559.85 | 550.59 | 551.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 558.85 | 552.25 | 552.52 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 563.95 | 554.59 | 553.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 567.00 | 557.07 | 554.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 569.25 | 572.63 | 566.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 13:15:00 | 569.25 | 572.63 | 566.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 569.25 | 572.63 | 566.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 566.35 | 572.63 | 566.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 570.70 | 571.79 | 567.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:00:00 | 578.30 | 573.09 | 568.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:45:00 | 580.00 | 575.30 | 570.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 578.20 | 576.66 | 572.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 13:45:00 | 577.80 | 578.73 | 575.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 572.20 | 577.43 | 574.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 572.20 | 577.43 | 574.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 572.05 | 576.35 | 574.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 573.70 | 576.35 | 574.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 578.65 | 576.46 | 574.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 567.00 | 573.50 | 573.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 567.00 | 573.50 | 573.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 561.25 | 571.05 | 572.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 13:15:00 | 534.95 | 532.64 | 540.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 14:00:00 | 534.95 | 532.64 | 540.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 549.00 | 534.98 | 539.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 551.55 | 534.98 | 539.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 549.75 | 537.94 | 540.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 549.85 | 537.94 | 540.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 549.00 | 542.00 | 541.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 14:15:00 | 553.20 | 544.72 | 543.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 10:15:00 | 543.25 | 545.27 | 543.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 10:15:00 | 543.25 | 545.27 | 543.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 543.25 | 545.27 | 543.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 543.80 | 545.27 | 543.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 542.95 | 544.81 | 543.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:15:00 | 543.40 | 544.81 | 543.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 544.00 | 544.65 | 543.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:00:00 | 546.20 | 544.96 | 544.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:45:00 | 547.00 | 544.74 | 544.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 551.40 | 544.79 | 544.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 09:15:00 | 600.82 | 592.51 | 587.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 534.00 | 578.87 | 584.47 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 571.40 | 565.42 | 564.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 574.85 | 567.31 | 565.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 583.00 | 587.11 | 584.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 583.00 | 587.11 | 584.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 583.00 | 587.11 | 584.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 586.20 | 587.11 | 584.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 570.10 | 583.71 | 583.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 564.35 | 583.71 | 583.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 577.00 | 582.37 | 582.85 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 583.55 | 582.98 | 582.92 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 578.20 | 582.57 | 582.88 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 587.20 | 583.50 | 583.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 12:15:00 | 590.90 | 585.88 | 584.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 590.25 | 592.04 | 589.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 590.25 | 592.04 | 589.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 590.25 | 592.04 | 589.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 594.05 | 592.04 | 589.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 586.15 | 594.59 | 594.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 586.15 | 594.59 | 594.98 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 10:15:00 | 604.50 | 595.55 | 594.58 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 591.25 | 595.80 | 595.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 583.35 | 589.16 | 592.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 588.00 | 587.45 | 590.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 588.00 | 587.45 | 590.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 588.00 | 587.45 | 590.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:15:00 | 589.25 | 587.45 | 590.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 587.20 | 587.40 | 589.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 586.45 | 587.12 | 589.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 592.65 | 586.83 | 587.10 | SL hit (close>static) qty=1.00 sl=591.35 alert=retest2 |

### Cycle 115 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 593.00 | 588.06 | 587.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 593.20 | 589.60 | 588.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 12:15:00 | 588.10 | 589.42 | 588.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 12:15:00 | 588.10 | 589.42 | 588.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 588.10 | 589.42 | 588.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 588.10 | 589.42 | 588.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 594.25 | 590.38 | 589.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:15:00 | 597.65 | 590.38 | 589.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 15:15:00 | 607.95 | 609.45 | 609.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 15:15:00 | 607.95 | 609.45 | 609.62 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 614.40 | 610.44 | 610.05 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 607.00 | 609.90 | 610.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 596.15 | 607.15 | 608.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 587.60 | 580.70 | 588.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 587.60 | 580.70 | 588.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 591.95 | 582.95 | 588.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 591.95 | 582.95 | 588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 588.40 | 584.04 | 588.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 583.65 | 584.99 | 588.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:30:00 | 582.80 | 579.05 | 579.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 11:15:00 | 591.05 | 581.84 | 580.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 591.05 | 581.84 | 580.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 596.25 | 584.72 | 582.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 590.30 | 592.02 | 587.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 590.30 | 592.02 | 587.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 589.25 | 592.77 | 589.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 589.25 | 592.77 | 589.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 588.00 | 591.82 | 589.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 581.85 | 591.82 | 589.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 578.55 | 589.17 | 588.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 578.55 | 589.17 | 588.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 577.35 | 586.80 | 587.34 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 15:15:00 | 590.80 | 587.03 | 587.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 600.55 | 589.74 | 588.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 582.40 | 593.54 | 591.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 582.40 | 593.54 | 591.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 582.40 | 593.54 | 591.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 582.40 | 593.54 | 591.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 590.90 | 593.01 | 591.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 599.50 | 593.01 | 591.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 580.65 | 589.09 | 590.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 580.65 | 589.09 | 590.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 554.70 | 579.85 | 585.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 530.40 | 527.57 | 543.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 530.40 | 527.57 | 543.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 561.50 | 537.84 | 540.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 561.50 | 537.84 | 540.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 556.85 | 541.64 | 542.36 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 552.65 | 543.85 | 543.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 564.55 | 552.08 | 547.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 12:15:00 | 558.10 | 558.34 | 552.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:00:00 | 558.10 | 558.34 | 552.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 555.70 | 557.29 | 553.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:30:00 | 558.95 | 557.29 | 553.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 560.40 | 557.65 | 554.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:00:00 | 567.35 | 557.26 | 554.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 569.35 | 559.76 | 558.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:30:00 | 567.05 | 563.62 | 560.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 567.45 | 576.56 | 576.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 567.45 | 576.56 | 576.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 563.95 | 574.04 | 575.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 556.85 | 544.47 | 551.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 556.85 | 544.47 | 551.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 556.85 | 544.47 | 551.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 556.85 | 544.47 | 551.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 552.60 | 546.09 | 551.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:15:00 | 551.10 | 546.09 | 551.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 550.15 | 546.91 | 551.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 523.54 | 536.95 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 522.64 | 536.95 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 523.70 | 520.43 | 529.97 | SL hit (close>ema200) qty=0.50 sl=520.43 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 534.95 | 521.99 | 521.98 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 517.65 | 524.78 | 525.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 515.15 | 522.86 | 524.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 09:15:00 | 516.85 | 514.36 | 518.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 516.85 | 514.36 | 518.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 516.85 | 514.36 | 518.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 478.70 | 509.68 | 512.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 454.76 | 468.03 | 484.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 468.65 | 466.32 | 480.59 | SL hit (close>ema200) qty=0.50 sl=466.32 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 485.50 | 474.85 | 474.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 494.65 | 478.81 | 476.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 490.50 | 490.58 | 485.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:15:00 | 489.55 | 490.58 | 485.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 494.20 | 491.69 | 488.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 490.85 | 491.69 | 488.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 489.05 | 491.19 | 488.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 489.10 | 491.19 | 488.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 479.35 | 488.82 | 488.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 479.35 | 488.82 | 488.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 479.75 | 487.01 | 487.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 11:15:00 | 474.90 | 480.87 | 483.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 480.65 | 479.70 | 482.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 480.65 | 479.70 | 482.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 474.95 | 478.81 | 481.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 473.65 | 478.81 | 481.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 472.85 | 477.99 | 480.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 474.15 | 475.12 | 478.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 471.25 | 475.74 | 478.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 478.10 | 476.21 | 478.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 470.00 | 473.84 | 476.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 483.95 | 475.25 | 476.55 | SL hit (close>static) qty=1.00 sl=483.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 486.45 | 478.75 | 478.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 487.75 | 481.82 | 479.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 495.40 | 496.29 | 492.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 495.40 | 496.29 | 492.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 501.90 | 502.04 | 498.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 498.50 | 502.04 | 498.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 501.30 | 507.45 | 504.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 504.05 | 507.45 | 504.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 507.30 | 507.42 | 504.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:45:00 | 510.70 | 507.39 | 505.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 498.55 | 503.49 | 504.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 498.55 | 503.49 | 504.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 495.35 | 501.87 | 503.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 492.05 | 488.00 | 493.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 492.05 | 488.00 | 493.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 492.05 | 488.00 | 493.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 492.05 | 488.00 | 493.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 493.60 | 489.12 | 493.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 494.75 | 489.12 | 493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 489.95 | 489.28 | 493.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 489.00 | 490.01 | 492.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:45:00 | 489.60 | 488.27 | 491.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 489.20 | 488.23 | 490.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:15:00 | 489.15 | 485.03 | 485.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 490.40 | 486.11 | 485.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 490.40 | 486.11 | 485.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 491.90 | 487.50 | 486.45 | Break + close above crossover candle high |

### Cycle 132 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 457.05 | 481.97 | 484.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 456.05 | 468.61 | 476.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 442.40 | 440.59 | 450.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 442.40 | 440.59 | 450.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 451.60 | 437.83 | 441.66 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 450.50 | 444.72 | 444.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 462.55 | 450.63 | 447.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 458.80 | 459.37 | 454.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 457.75 | 460.43 | 457.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 457.75 | 460.43 | 457.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 457.75 | 460.43 | 457.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 461.35 | 460.62 | 457.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 466.00 | 461.65 | 458.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 460.80 | 475.98 | 477.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 460.80 | 475.98 | 477.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 453.45 | 471.47 | 475.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 463.75 | 460.43 | 467.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 463.75 | 460.43 | 467.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 457.85 | 460.59 | 464.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 455.20 | 458.67 | 462.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:30:00 | 454.80 | 458.51 | 461.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 454.40 | 458.51 | 461.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 455.10 | 458.39 | 460.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 462.05 | 457.50 | 459.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 464.50 | 457.50 | 459.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 459.15 | 457.83 | 459.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 463.50 | 457.83 | 459.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 453.20 | 456.91 | 458.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 451.00 | 456.91 | 458.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 460.30 | 456.79 | 457.76 | SL hit (close>static) qty=1.00 sl=459.20 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 461.60 | 458.49 | 458.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 462.70 | 460.03 | 459.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 453.85 | 459.15 | 458.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 453.85 | 459.15 | 458.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 453.85 | 459.15 | 458.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 453.85 | 459.15 | 458.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 455.45 | 458.41 | 458.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 451.85 | 457.10 | 458.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 451.60 | 450.38 | 452.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 451.60 | 450.38 | 452.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 453.60 | 451.02 | 452.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 453.60 | 451.02 | 452.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 452.45 | 451.31 | 452.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 452.35 | 451.31 | 452.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 454.65 | 451.98 | 453.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 453.30 | 451.98 | 453.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 452.45 | 452.07 | 452.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:00:00 | 449.45 | 451.55 | 452.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 451.50 | 446.34 | 446.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 451.50 | 446.34 | 446.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 455.75 | 448.22 | 447.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 497.85 | 497.95 | 488.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 497.85 | 497.95 | 488.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 508.30 | 512.11 | 508.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 508.25 | 512.11 | 508.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 510.50 | 511.79 | 508.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 518.65 | 511.79 | 508.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 526.40 | 527.83 | 528.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 526.40 | 527.83 | 528.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 524.35 | 527.13 | 527.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 525.45 | 524.31 | 525.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 524.80 | 524.41 | 525.65 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 530.65 | 526.70 | 526.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 535.65 | 528.49 | 527.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 532.60 | 532.65 | 530.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:15:00 | 528.95 | 532.65 | 530.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 528.20 | 531.76 | 530.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 529.85 | 531.76 | 530.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 526.60 | 530.73 | 529.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 526.60 | 530.73 | 529.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 526.00 | 529.78 | 529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 526.00 | 529.78 | 529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 529.75 | 529.63 | 529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 529.75 | 529.63 | 529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 531.00 | 529.91 | 529.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 533.30 | 529.91 | 529.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 528.90 | 529.70 | 529.46 | SL hit (close<static) qty=1.00 sl=529.25 alert=retest2 |

### Cycle 140 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 532.80 | 537.05 | 537.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 530.40 | 535.72 | 536.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 512.15 | 510.87 | 518.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 512.15 | 510.87 | 518.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 512.40 | 513.44 | 517.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 509.45 | 511.85 | 515.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 483.98 | 490.52 | 497.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 487.80 | 485.55 | 491.43 | SL hit (close>ema200) qty=0.50 sl=485.55 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 498.40 | 491.90 | 491.46 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 486.95 | 491.77 | 492.38 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 498.25 | 493.86 | 493.28 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 491.25 | 493.09 | 493.12 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 496.35 | 493.28 | 493.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 496.75 | 493.97 | 493.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 499.15 | 497.45 | 496.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 497.85 | 497.53 | 496.32 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 494.75 | 495.56 | 495.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 487.90 | 493.99 | 494.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 489.25 | 489.18 | 491.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 489.25 | 489.18 | 491.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 490.25 | 489.29 | 491.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 488.10 | 490.55 | 491.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 489.05 | 490.27 | 491.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 487.85 | 490.48 | 491.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 493.05 | 491.75 | 491.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 493.05 | 491.75 | 491.67 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 490.60 | 491.75 | 491.76 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 495.30 | 492.46 | 492.08 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 486.05 | 491.82 | 492.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 478.55 | 489.17 | 491.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 485.00 | 483.28 | 486.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 474.00 | 471.47 | 474.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 474.20 | 471.47 | 474.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 475.85 | 472.35 | 474.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 476.80 | 472.35 | 474.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 474.80 | 472.84 | 474.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:15:00 | 478.25 | 472.84 | 474.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 481.05 | 474.48 | 475.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 478.85 | 474.48 | 475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 485.60 | 476.70 | 476.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 489.80 | 484.26 | 480.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 499.65 | 502.06 | 497.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 499.65 | 502.06 | 497.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 497.60 | 501.17 | 497.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 497.05 | 501.17 | 497.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 498.90 | 500.72 | 497.77 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 495.55 | 496.92 | 496.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 492.85 | 495.61 | 496.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 482.30 | 478.03 | 482.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 478.00 | 478.03 | 481.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 481.55 | 478.03 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 476.75 | 472.21 | 474.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 473.45 | 472.21 | 474.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 470.85 | 471.93 | 474.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 470.15 | 471.58 | 474.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 469.50 | 468.39 | 471.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 480.65 | 472.97 | 472.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 480.65 | 472.97 | 472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 484.00 | 475.17 | 473.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 480.60 | 487.26 | 483.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 479.75 | 485.76 | 483.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:15:00 | 480.00 | 485.76 | 483.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 479.75 | 484.56 | 482.73 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 474.05 | 480.80 | 481.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 462.65 | 476.32 | 479.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 459.25 | 456.71 | 463.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 459.70 | 456.71 | 463.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 446.75 | 441.87 | 446.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 448.50 | 441.87 | 446.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 442.45 | 441.98 | 446.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 441.15 | 441.98 | 446.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 450.75 | 442.88 | 445.20 | SL hit (close>static) qty=1.00 sl=447.70 alert=retest2 |

### Cycle 155 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 450.80 | 443.90 | 443.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 457.00 | 448.34 | 445.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 459.55 | 460.52 | 456.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:45:00 | 458.80 | 460.52 | 456.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 460.45 | 460.50 | 456.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 456.25 | 460.50 | 456.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 459.10 | 459.98 | 457.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 458.85 | 459.98 | 457.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 460.25 | 460.04 | 457.77 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 457.95 | 458.49 | 458.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 453.80 | 457.55 | 458.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 458.25 | 457.54 | 457.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 459.20 | 457.87 | 458.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 459.20 | 457.87 | 458.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 461.65 | 458.84 | 458.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 463.75 | 460.98 | 460.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 459.40 | 460.81 | 460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 456.80 | 460.01 | 460.06 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 465.90 | 461.19 | 460.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 471.00 | 464.40 | 462.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 467.55 | 468.34 | 465.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 470.45 | 468.34 | 465.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 473.05 | 469.28 | 466.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 484.60 | 470.23 | 468.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 11:15:00 | 533.06 | 526.34 | 524.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 533.60 | 539.57 | 540.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 529.45 | 537.55 | 539.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 531.80 | 525.45 | 529.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 531.30 | 527.01 | 529.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 531.30 | 527.01 | 529.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 532.10 | 528.03 | 529.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 532.10 | 528.03 | 529.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 531.50 | 529.57 | 530.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 531.45 | 529.57 | 530.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 532.15 | 530.51 | 530.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 531.15 | 529.84 | 530.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 516.95 | 530.08 | 530.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 531.35 | 522.41 | 521.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 531.35 | 522.41 | 521.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 537.30 | 527.24 | 523.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 14:15:00 | 552.30 | 553.85 | 547.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 552.30 | 553.85 | 547.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 546.00 | 551.35 | 547.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 546.00 | 551.35 | 547.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 555.55 | 552.19 | 548.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 558.00 | 553.41 | 550.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 562.45 | 551.41 | 550.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 567.10 | 569.04 | 569.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 567.10 | 569.04 | 569.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 564.15 | 568.06 | 568.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 571.00 | 567.19 | 567.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 569.95 | 567.75 | 567.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 569.95 | 567.75 | 567.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 566.45 | 567.58 | 567.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 569.15 | 567.58 | 567.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 567.85 | 567.64 | 567.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:00:00 | 562.05 | 566.52 | 567.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 566.40 | 562.65 | 562.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 566.40 | 562.65 | 562.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 576.10 | 569.33 | 566.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 571.05 | 573.39 | 569.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 572.00 | 573.11 | 570.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 570.75 | 573.11 | 570.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 572.80 | 572.70 | 570.39 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 569.25 | 570.84 | 570.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 565.20 | 569.71 | 570.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 572.25 | 569.66 | 570.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 569.75 | 569.68 | 570.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 569.75 | 569.68 | 570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 571.55 | 570.05 | 570.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 571.55 | 570.05 | 570.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 571.15 | 570.27 | 570.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 570.55 | 570.27 | 570.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 573.00 | 570.82 | 570.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 573.00 | 570.82 | 570.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 15:15:00 | 576.10 | 572.36 | 571.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 572.30 | 572.35 | 571.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 568.30 | 571.54 | 571.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 568.30 | 571.54 | 571.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 567.75 | 570.78 | 570.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 566.90 | 568.96 | 569.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 554.50 | 554.22 | 558.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 552.80 | 554.22 | 558.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 552.10 | 554.03 | 557.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:15:00 | 550.70 | 554.03 | 557.74 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 550.40 | 552.05 | 555.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:30:00 | 553.10 | 552.05 | 555.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 537.20 | 545.93 | 551.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 535.00 | 542.16 | 548.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 535.00 | 540.97 | 547.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 543.30 | 540.89 | 542.98 | SL hit (close>ema400) qty=1.00 sl=542.98 alert=retest1 |

### Cycle 167 — BUY (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 14:15:00 | 556.05 | 546.39 | 545.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 562.55 | 553.33 | 550.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 10:15:00 | 555.10 | 555.16 | 551.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:30:00 | 553.00 | 555.16 | 551.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 559.65 | 555.54 | 552.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:45:00 | 561.00 | 556.62 | 553.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 552.15 | 553.72 | 553.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 552.15 | 553.72 | 553.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 550.70 | 552.68 | 553.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 540.40 | 536.78 | 541.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 543.05 | 538.03 | 541.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 543.05 | 538.03 | 541.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 540.90 | 538.60 | 541.25 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 550.95 | 544.18 | 543.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 554.05 | 548.95 | 546.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 552.80 | 550.11 | 547.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 560.00 | 566.37 | 566.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 560.00 | 566.37 | 566.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 552.90 | 562.55 | 564.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 561.15 | 560.66 | 563.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 15:00:00 | 561.15 | 560.66 | 563.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 562.35 | 561.37 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 565.90 | 561.37 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 557.20 | 560.53 | 562.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 563.85 | 560.53 | 562.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 554.65 | 551.25 | 555.64 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 560.45 | 558.15 | 557.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 564.25 | 559.37 | 558.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 556.70 | 558.83 | 558.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 556.35 | 558.34 | 558.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 558.35 | 558.34 | 558.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 12:15:00 | 556.20 | 557.91 | 557.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 556.20 | 557.91 | 557.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 552.60 | 556.59 | 557.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 561.40 | 557.56 | 557.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 562.00 | 558.44 | 558.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 566.50 | 560.06 | 558.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 571.75 | 571.86 | 568.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:30:00 | 571.40 | 571.86 | 568.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 569.75 | 571.07 | 569.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 570.20 | 571.07 | 569.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 569.15 | 570.69 | 569.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 569.15 | 570.69 | 569.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 574.45 | 571.44 | 569.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 576.70 | 574.64 | 572.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 575.95 | 574.67 | 572.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 577.00 | 574.16 | 572.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:00:00 | 577.05 | 576.25 | 574.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 581.85 | 582.55 | 579.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 580.25 | 582.55 | 579.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 587.50 | 584.26 | 581.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 581.80 | 584.26 | 581.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 608.85 | 614.75 | 611.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 608.85 | 614.75 | 611.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 606.60 | 613.12 | 610.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 606.60 | 613.12 | 610.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 604.55 | 608.65 | 609.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 604.55 | 608.65 | 609.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 597.90 | 603.67 | 606.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 608.20 | 598.88 | 600.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 611.90 | 601.48 | 601.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 612.65 | 601.48 | 601.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 612.85 | 603.76 | 602.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 619.45 | 611.58 | 607.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 15:15:00 | 615.00 | 615.21 | 612.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 610.10 | 615.21 | 612.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 611.35 | 614.44 | 612.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 610.40 | 614.44 | 612.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 610.20 | 613.59 | 611.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 610.45 | 613.59 | 611.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 611.45 | 613.16 | 611.80 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 605.55 | 610.78 | 611.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 604.25 | 609.48 | 610.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 609.70 | 609.25 | 610.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 608.25 | 609.05 | 609.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 610.20 | 609.05 | 609.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 613.50 | 609.94 | 610.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 613.50 | 609.94 | 610.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 610.75 | 610.10 | 610.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 612.30 | 610.10 | 610.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 612.90 | 610.66 | 610.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 624.70 | 613.39 | 611.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 615.35 | 615.66 | 613.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 614.80 | 615.49 | 613.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 614.30 | 615.49 | 613.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 613.00 | 614.99 | 613.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:30:00 | 614.35 | 614.99 | 613.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 610.50 | 614.09 | 613.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 613.25 | 615.18 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 610.15 | 614.17 | 613.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 610.15 | 614.17 | 613.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 610.45 | 613.43 | 613.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 604.20 | 613.43 | 613.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 608.55 | 612.45 | 612.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 606.00 | 609.90 | 611.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 614.55 | 609.99 | 611.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 610.20 | 610.03 | 611.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 609.25 | 609.10 | 610.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 578.79 | 595.42 | 602.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 595.05 | 594.20 | 600.04 | SL hit (close>ema200) qty=0.50 sl=594.20 alert=retest2 |

### Cycle 179 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 571.00 | 560.42 | 560.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 579.75 | 566.14 | 563.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 564.35 | 571.38 | 567.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 563.15 | 569.73 | 567.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 571.80 | 569.73 | 567.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 12:15:00 | 564.90 | 569.69 | 570.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 564.90 | 569.69 | 570.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 558.15 | 565.27 | 567.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 566.30 | 563.65 | 566.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 568.35 | 564.59 | 566.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 568.35 | 564.59 | 566.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 574.35 | 566.54 | 567.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 574.35 | 566.54 | 567.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 574.30 | 568.09 | 567.76 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 561.40 | 567.40 | 568.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 557.85 | 565.49 | 567.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 553.45 | 549.32 | 555.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 553.45 | 549.32 | 555.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 553.00 | 550.06 | 555.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 572.80 | 550.06 | 555.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 566.85 | 553.42 | 556.14 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 569.35 | 558.99 | 558.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 570.75 | 561.34 | 559.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 577.50 | 579.11 | 573.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 577.50 | 579.11 | 573.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 567.25 | 576.81 | 574.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 567.25 | 576.81 | 574.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 563.95 | 574.23 | 573.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 562.85 | 574.23 | 573.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 560.65 | 571.52 | 572.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 557.65 | 568.74 | 571.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 574.10 | 567.85 | 569.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 583.10 | 570.90 | 571.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 583.10 | 570.90 | 571.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 578.70 | 572.46 | 571.71 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 15:15:00 | 569.00 | 570.99 | 571.22 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 580.10 | 572.81 | 572.03 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 564.90 | 574.32 | 575.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 562.45 | 571.95 | 573.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 15:15:00 | 570.00 | 569.83 | 572.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 563.55 | 569.83 | 572.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 558.00 | 563.89 | 567.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 566.20 | 564.01 | 565.29 | SL hit (close>ema400) qty=1.00 sl=565.29 alert=retest1 |

### Cycle 189 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 569.00 | 566.42 | 566.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 573.70 | 568.21 | 567.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 576.25 | 577.17 | 573.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 576.25 | 577.17 | 573.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 575.00 | 577.29 | 575.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 572.25 | 577.29 | 575.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 575.85 | 577.00 | 575.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 577.40 | 577.00 | 575.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 576.30 | 577.02 | 575.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 576.00 | 577.02 | 575.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 596.20 | 598.55 | 596.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 595.80 | 598.55 | 596.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 594.35 | 597.71 | 596.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 594.35 | 597.71 | 596.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 591.45 | 596.46 | 595.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 591.90 | 596.46 | 595.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 591.65 | 594.85 | 595.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 15:15:00 | 586.35 | 593.15 | 594.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 583.25 | 571.72 | 570.99 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 555.95 | 570.49 | 572.03 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 577.45 | 570.26 | 569.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 580.20 | 573.88 | 571.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 588.15 | 584.69 | 580.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 588.65 | 585.18 | 582.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 568.65 | 581.85 | 581.22 | SL hit (close<static) qty=1.00 sl=572.20 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 562.80 | 578.04 | 579.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 560.20 | 574.47 | 577.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 570.25 | 567.05 | 572.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 561.20 | 564.75 | 569.65 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 571.95 | 569.36 | 569.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 575.70 | 570.76 | 569.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 570.45 | 579.00 | 576.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 569.45 | 577.09 | 575.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 569.45 | 577.09 | 575.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 582.15 | 578.10 | 576.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 588.15 | 578.48 | 576.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 12:15:00 | 568.00 | 578.45 | 579.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 568.00 | 578.45 | 579.73 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 585.00 | 580.57 | 580.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 593.90 | 583.23 | 581.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 15:15:00 | 620.00 | 622.75 | 614.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 09:15:00 | 622.75 | 622.75 | 614.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 621.15 | 622.76 | 617.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 621.15 | 622.76 | 617.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 620.10 | 622.26 | 618.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 637.25 | 622.26 | 618.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 606.50 | 623.55 | 622.41 | SL hit (close<static) qty=1.00 sl=616.80 alert=retest2 |

### Cycle 198 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 612.50 | 621.34 | 621.51 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 628.20 | 620.72 | 620.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 632.60 | 623.10 | 621.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 631.00 | 632.17 | 628.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:45:00 | 630.55 | 632.17 | 628.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 636.30 | 639.93 | 635.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 635.25 | 639.93 | 635.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 639.55 | 639.85 | 636.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 642.15 | 636.95 | 636.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 13:15:00 | 633.60 | 638.70 | 637.67 | SL hit (close<static) qty=1.00 sl=636.20 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 631.95 | 636.28 | 636.68 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 642.00 | 637.09 | 636.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 649.40 | 639.55 | 637.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 642.75 | 646.82 | 642.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 645.40 | 646.27 | 643.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 640.55 | 646.27 | 643.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 642.85 | 645.59 | 643.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:15:00 | 648.00 | 645.47 | 643.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-29 09:15:00 | 712.80 | 704.23 | 698.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 696.00 | 702.65 | 702.75 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 706.50 | 703.42 | 703.10 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 701.10 | 703.16 | 703.27 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 706.50 | 703.94 | 703.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 707.30 | 704.62 | 703.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 750.10 | 750.17 | 742.57 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 15:15:00 | 284.85 | 2023-05-24 15:15:00 | 279.50 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2023-05-17 13:30:00 | 285.80 | 2023-05-24 15:15:00 | 279.50 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2023-05-18 10:00:00 | 286.40 | 2023-05-24 15:15:00 | 279.50 | STOP_HIT | 1.00 | 2.41% |
| SELL | retest2 | 2023-05-18 10:30:00 | 287.05 | 2023-05-24 15:15:00 | 279.50 | STOP_HIT | 1.00 | 2.63% |
| BUY | retest2 | 2023-05-31 12:45:00 | 283.95 | 2023-06-06 09:15:00 | 281.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-05-31 14:15:00 | 283.80 | 2023-06-06 09:15:00 | 281.30 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-06-01 09:15:00 | 286.00 | 2023-06-06 09:15:00 | 281.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-06-07 12:00:00 | 283.50 | 2023-06-07 15:15:00 | 284.25 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-06-07 13:45:00 | 283.55 | 2023-06-07 15:15:00 | 284.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-07-13 09:15:00 | 310.25 | 2023-07-13 12:15:00 | 305.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-07-13 10:00:00 | 310.65 | 2023-07-13 12:15:00 | 305.90 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-07-13 11:00:00 | 310.30 | 2023-07-13 12:15:00 | 305.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-07-20 13:45:00 | 314.65 | 2023-07-20 14:15:00 | 311.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-07-21 10:00:00 | 313.40 | 2023-07-21 11:15:00 | 311.45 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-08-03 12:15:00 | 318.15 | 2023-08-07 11:15:00 | 321.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-08-04 15:00:00 | 318.10 | 2023-08-07 11:15:00 | 321.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-08-08 13:15:00 | 324.40 | 2023-08-10 09:15:00 | 310.60 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2023-08-09 09:15:00 | 324.70 | 2023-08-10 09:15:00 | 310.60 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2023-08-09 11:30:00 | 324.00 | 2023-08-10 09:15:00 | 310.60 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2023-08-18 12:45:00 | 289.95 | 2023-08-21 14:15:00 | 298.45 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2023-08-18 15:15:00 | 290.00 | 2023-08-21 14:15:00 | 298.45 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2023-08-21 10:00:00 | 290.35 | 2023-08-21 14:15:00 | 298.45 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-08-29 09:15:00 | 301.00 | 2023-08-31 14:15:00 | 297.65 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-09-11 09:15:00 | 316.45 | 2023-09-12 09:15:00 | 310.85 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2023-09-27 10:45:00 | 345.05 | 2023-10-09 15:15:00 | 354.25 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2023-09-28 10:15:00 | 345.70 | 2023-10-09 15:15:00 | 354.25 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2023-09-28 12:15:00 | 347.30 | 2023-10-09 15:15:00 | 354.25 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2023-09-28 13:45:00 | 345.70 | 2023-10-09 15:15:00 | 354.25 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2023-09-29 09:15:00 | 344.70 | 2023-10-09 15:15:00 | 354.25 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2023-10-13 15:15:00 | 352.40 | 2023-10-18 09:15:00 | 356.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-10-16 11:00:00 | 353.10 | 2023-10-18 09:15:00 | 356.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-10-16 14:15:00 | 352.80 | 2023-10-18 09:15:00 | 356.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-10-17 10:00:00 | 352.15 | 2023-10-18 09:15:00 | 356.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2023-10-17 15:15:00 | 353.45 | 2023-10-18 09:15:00 | 356.95 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-10-31 10:15:00 | 321.95 | 2023-11-01 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2023-11-13 10:45:00 | 370.45 | 2023-11-20 13:15:00 | 365.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-11-15 09:15:00 | 370.60 | 2023-11-20 13:15:00 | 365.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-11-17 09:15:00 | 372.20 | 2023-11-20 13:15:00 | 365.20 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-11-17 11:00:00 | 374.15 | 2023-11-20 13:15:00 | 365.20 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2023-11-29 09:15:00 | 382.85 | 2023-12-05 09:15:00 | 389.60 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2023-12-12 12:45:00 | 389.15 | 2023-12-15 09:15:00 | 391.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-12-12 14:15:00 | 387.50 | 2023-12-15 09:15:00 | 391.45 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-12-22 12:30:00 | 388.15 | 2023-12-26 13:15:00 | 391.15 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-12-22 13:00:00 | 385.75 | 2023-12-26 13:15:00 | 391.15 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-12-29 12:45:00 | 401.50 | 2024-01-05 13:15:00 | 412.80 | STOP_HIT | 1.00 | 2.81% |
| SELL | retest2 | 2024-01-10 12:00:00 | 404.10 | 2024-01-11 15:15:00 | 411.35 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-01-10 12:45:00 | 402.00 | 2024-01-11 15:15:00 | 411.35 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-01-15 10:15:00 | 419.90 | 2024-01-18 09:15:00 | 416.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-01-23 14:15:00 | 413.75 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-01-24 10:00:00 | 415.15 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-01-24 11:00:00 | 415.40 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-01-25 10:45:00 | 414.05 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-01-29 10:30:00 | 409.00 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-01-29 13:15:00 | 410.05 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-01-30 10:30:00 | 410.40 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-01-30 15:00:00 | 409.20 | 2024-01-31 12:15:00 | 416.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-02-14 10:45:00 | 428.05 | 2024-02-19 09:15:00 | 470.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-14 13:30:00 | 428.90 | 2024-02-19 09:15:00 | 471.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-02 09:15:00 | 467.75 | 2024-03-06 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-03-04 09:15:00 | 465.55 | 2024-03-06 09:15:00 | 456.35 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-03-05 09:45:00 | 466.00 | 2024-03-06 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-03-05 13:00:00 | 465.80 | 2024-03-06 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-03-11 09:30:00 | 454.85 | 2024-03-12 10:15:00 | 432.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:30:00 | 454.85 | 2024-03-13 14:15:00 | 409.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-28 11:00:00 | 430.70 | 2024-04-08 12:15:00 | 442.25 | STOP_HIT | 1.00 | 2.68% |
| BUY | retest2 | 2024-03-28 15:15:00 | 431.75 | 2024-04-08 12:15:00 | 442.25 | STOP_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2024-04-12 11:30:00 | 430.00 | 2024-04-15 13:15:00 | 408.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 11:30:00 | 430.00 | 2024-04-16 10:15:00 | 416.65 | STOP_HIT | 0.50 | 3.10% |
| BUY | retest2 | 2024-04-29 12:00:00 | 422.05 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-04-30 09:15:00 | 422.75 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-04-30 10:00:00 | 425.80 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-02 10:45:00 | 423.00 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-05-02 14:30:00 | 425.75 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-02 15:15:00 | 425.80 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-03 10:30:00 | 425.25 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-05-03 12:15:00 | 425.10 | 2024-05-06 09:15:00 | 418.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-05-15 10:15:00 | 396.40 | 2024-05-15 12:15:00 | 404.95 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-05-15 14:00:00 | 395.10 | 2024-05-17 09:15:00 | 402.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-05-15 15:15:00 | 396.00 | 2024-05-17 09:15:00 | 402.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-05-16 11:15:00 | 395.80 | 2024-05-17 09:15:00 | 402.40 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-05-28 09:15:00 | 444.50 | 2024-05-29 12:15:00 | 430.85 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-06-25 09:15:00 | 497.15 | 2024-06-28 14:15:00 | 491.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-06-26 09:15:00 | 492.40 | 2024-07-01 09:15:00 | 491.80 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-06-26 10:00:00 | 491.45 | 2024-07-01 09:15:00 | 491.80 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-06-26 13:30:00 | 492.00 | 2024-07-01 09:15:00 | 491.80 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-06-28 10:45:00 | 503.60 | 2024-07-01 09:15:00 | 491.80 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-09 12:15:00 | 522.00 | 2024-07-10 09:15:00 | 512.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-07-30 12:15:00 | 577.55 | 2024-08-02 09:15:00 | 635.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 10:15:00 | 689.05 | 2024-08-27 09:15:00 | 679.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-06 10:15:00 | 706.75 | 2024-09-06 11:15:00 | 699.10 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-09-25 13:45:00 | 553.60 | 2024-09-26 12:15:00 | 543.30 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-16 09:15:00 | 608.00 | 2024-10-17 13:15:00 | 594.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-10-16 10:30:00 | 604.50 | 2024-10-17 13:15:00 | 594.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-10-16 12:30:00 | 603.00 | 2024-10-17 13:15:00 | 594.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-10-17 10:45:00 | 602.10 | 2024-10-17 13:15:00 | 594.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-10-21 15:00:00 | 589.50 | 2024-10-22 14:15:00 | 560.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 587.90 | 2024-10-22 14:15:00 | 558.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:00:00 | 589.50 | 2024-10-23 10:15:00 | 574.35 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2024-10-22 10:00:00 | 587.90 | 2024-10-23 10:15:00 | 574.35 | STOP_HIT | 0.50 | 2.30% |
| BUY | retest2 | 2024-11-08 11:00:00 | 578.30 | 2024-11-12 13:15:00 | 567.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-11-08 14:45:00 | 580.00 | 2024-11-12 13:15:00 | 567.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-11-11 09:45:00 | 578.20 | 2024-11-12 13:15:00 | 567.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-11-11 13:45:00 | 577.80 | 2024-11-12 13:15:00 | 567.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-11-21 14:00:00 | 546.20 | 2024-12-03 09:15:00 | 600.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-21 14:45:00 | 547.00 | 2024-12-03 09:15:00 | 601.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 09:15:00 | 551.40 | 2024-12-03 12:15:00 | 606.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-19 10:15:00 | 594.05 | 2024-12-23 09:15:00 | 586.15 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-12-30 11:30:00 | 586.45 | 2024-12-31 13:15:00 | 592.65 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-01-01 14:15:00 | 597.65 | 2025-01-08 15:15:00 | 607.95 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2025-01-14 15:00:00 | 583.65 | 2025-01-20 11:15:00 | 591.05 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-01-17 10:30:00 | 582.80 | 2025-01-20 11:15:00 | 591.05 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-01-24 11:15:00 | 599.50 | 2025-01-24 13:15:00 | 580.65 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-02-01 14:00:00 | 567.35 | 2025-02-10 10:15:00 | 567.45 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-02-04 09:15:00 | 569.35 | 2025-02-10 10:15:00 | 567.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-02-04 14:30:00 | 567.05 | 2025-02-10 10:15:00 | 567.45 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-02-13 11:15:00 | 551.10 | 2025-02-14 10:15:00 | 523.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:00:00 | 550.15 | 2025-02-14 10:15:00 | 522.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 551.10 | 2025-02-17 10:15:00 | 523.70 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2025-02-13 12:00:00 | 550.15 | 2025-02-17 10:15:00 | 523.70 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2025-02-28 09:15:00 | 478.70 | 2025-03-03 10:15:00 | 454.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 478.70 | 2025-03-03 12:15:00 | 468.65 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2025-03-12 10:15:00 | 473.65 | 2025-03-17 09:15:00 | 483.95 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-03-12 11:15:00 | 472.85 | 2025-03-17 09:15:00 | 483.95 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-12 13:30:00 | 474.15 | 2025-03-17 09:15:00 | 483.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-03-13 09:15:00 | 471.25 | 2025-03-17 09:15:00 | 483.95 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-03-13 15:15:00 | 470.00 | 2025-03-17 09:15:00 | 483.95 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-03-25 11:45:00 | 510.70 | 2025-03-26 11:15:00 | 498.55 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-03-28 12:15:00 | 489.00 | 2025-04-03 11:15:00 | 490.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-03-28 14:45:00 | 489.60 | 2025-04-03 11:15:00 | 490.40 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-04-01 10:15:00 | 489.20 | 2025-04-03 11:15:00 | 490.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-04-03 11:15:00 | 489.15 | 2025-04-03 11:15:00 | 490.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-04-17 11:30:00 | 466.00 | 2025-04-25 09:15:00 | 460.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-29 13:00:00 | 455.20 | 2025-05-05 09:15:00 | 460.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-29 13:30:00 | 454.80 | 2025-05-05 11:15:00 | 461.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-04-29 14:15:00 | 454.40 | 2025-05-05 11:15:00 | 461.60 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-04-30 13:00:00 | 455.10 | 2025-05-05 11:15:00 | 461.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-02 12:15:00 | 451.00 | 2025-05-05 11:15:00 | 461.60 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-05-08 12:00:00 | 449.45 | 2025-05-12 11:15:00 | 451.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-21 09:15:00 | 518.65 | 2025-06-03 10:15:00 | 526.40 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-06-09 09:15:00 | 533.30 | 2025-06-09 09:15:00 | 528.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-09 11:00:00 | 531.25 | 2025-06-12 10:15:00 | 532.80 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-06-17 10:45:00 | 509.45 | 2025-06-19 12:15:00 | 483.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:45:00 | 509.45 | 2025-06-20 11:15:00 | 487.80 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-07-02 13:15:00 | 488.10 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-02 14:15:00 | 489.05 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-03 09:15:00 | 487.85 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-28 12:00:00 | 470.15 | 2025-07-29 12:15:00 | 480.65 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-07-29 10:15:00 | 469.50 | 2025-07-29 12:15:00 | 480.65 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-07 11:15:00 | 441.15 | 2025-08-07 14:15:00 | 450.75 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-08 10:30:00 | 440.55 | 2025-08-12 09:15:00 | 444.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-08 14:45:00 | 440.55 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-11 09:30:00 | 440.60 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-08-11 15:00:00 | 440.30 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-08-29 09:15:00 | 484.60 | 2025-09-12 11:15:00 | 533.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-25 11:30:00 | 531.15 | 2025-10-01 10:15:00 | 531.35 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-26 09:15:00 | 516.95 | 2025-10-01 10:15:00 | 531.35 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-10-07 15:15:00 | 558.00 | 2025-10-17 15:15:00 | 567.10 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2025-10-09 09:15:00 | 562.45 | 2025-10-17 15:15:00 | 567.10 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-10-23 13:00:00 | 562.05 | 2025-10-27 12:15:00 | 566.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-03 13:15:00 | 570.55 | 2025-11-03 13:15:00 | 573.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-10 09:15:00 | 552.80 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest1 | 2025-11-10 09:45:00 | 552.10 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest1 | 2025-11-10 10:15:00 | 550.70 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-11-11 14:00:00 | 535.00 | 2025-11-13 13:15:00 | 555.15 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-11-11 15:15:00 | 535.00 | 2025-11-13 13:15:00 | 555.15 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-11-18 14:45:00 | 561.00 | 2025-11-20 11:15:00 | 552.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-28 09:30:00 | 552.80 | 2025-12-05 09:15:00 | 560.00 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-12-10 12:15:00 | 558.35 | 2025-12-10 12:15:00 | 556.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-17 09:30:00 | 576.70 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.83% |
| BUY | retest2 | 2025-12-17 12:15:00 | 575.95 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-17 14:15:00 | 577.00 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-12-18 10:00:00 | 577.05 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.77% |
| SELL | retest2 | 2026-01-09 11:30:00 | 609.25 | 2026-01-12 11:15:00 | 578.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 609.25 | 2026-01-12 14:15:00 | 595.05 | STOP_HIT | 0.50 | 2.33% |
| BUY | retest2 | 2026-01-27 09:15:00 | 571.80 | 2026-01-28 12:15:00 | 564.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest1 | 2026-02-13 09:15:00 | 563.55 | 2026-02-17 11:15:00 | 566.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-03-12 11:00:00 | 588.15 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-12 15:00:00 | 588.65 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-20 09:15:00 | 588.15 | 2026-03-23 12:15:00 | 568.00 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-04-01 09:15:00 | 637.25 | 2026-04-02 09:15:00 | 606.50 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2026-04-10 09:15:00 | 642.15 | 2026-04-10 13:15:00 | 633.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-04-16 14:15:00 | 648.00 | 2026-04-29 09:15:00 | 712.80 | TARGET_HIT | 1.00 | 10.00% |
