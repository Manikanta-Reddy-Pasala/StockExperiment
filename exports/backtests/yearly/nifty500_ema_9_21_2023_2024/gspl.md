# Gujarat State Petronet Ltd. (GSPL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 289.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 239 |
| ALERT1 | 139 |
| ALERT2 | 135 |
| ALERT2_SKIP | 69 |
| ALERT3 | 358 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 167 |
| PARTIAL | 18 |
| TARGET_HIT | 9 |
| STOP_HIT | 159 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 186 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 74 / 112
- **Target hits / Stop hits / Partials:** 9 / 159 / 18
- **Avg / median % per leg:** 0.54% / -0.83%
- **Sum % (uncompounded):** 99.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 19 | 22.9% | 5 | 78 | 0 | -0.39% | -32.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.78% | -1.8% |
| BUY @ 3rd Alert (retest2) | 82 | 19 | 23.2% | 5 | 77 | 0 | -0.37% | -30.4% |
| SELL (all) | 103 | 55 | 53.4% | 4 | 81 | 18 | 1.28% | 131.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 55 | 53.4% | 4 | 81 | 18 | 1.28% | 131.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.78% | -1.8% |
| retest2 (combined) | 185 | 74 | 40.0% | 9 | 158 | 18 | 0.55% | 101.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 15:15:00 | 283.00 | 285.51 | 285.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 10:15:00 | 279.45 | 284.04 | 284.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 286.70 | 282.67 | 283.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 286.70 | 282.67 | 283.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 286.70 | 282.67 | 283.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:00:00 | 286.70 | 282.67 | 283.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 287.40 | 283.62 | 283.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:30:00 | 287.75 | 283.62 | 283.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 284.45 | 283.86 | 284.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:30:00 | 284.55 | 283.86 | 284.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 285.05 | 284.10 | 284.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 13:30:00 | 285.25 | 284.10 | 284.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 14:15:00 | 285.30 | 284.34 | 284.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 09:15:00 | 287.50 | 285.12 | 284.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 10:15:00 | 283.60 | 284.82 | 284.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 283.60 | 284.82 | 284.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 283.60 | 284.82 | 284.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 11:00:00 | 283.60 | 284.82 | 284.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 282.95 | 284.45 | 284.38 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 12:15:00 | 282.90 | 284.14 | 284.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 278.95 | 282.81 | 283.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 11:15:00 | 280.65 | 279.96 | 281.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 11:45:00 | 280.50 | 279.96 | 281.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 282.40 | 280.62 | 281.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 282.40 | 280.62 | 281.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 282.70 | 281.03 | 281.73 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 283.00 | 282.19 | 282.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 285.60 | 283.15 | 282.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 09:15:00 | 287.00 | 287.73 | 285.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 10:15:00 | 285.45 | 287.73 | 285.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 285.85 | 287.35 | 285.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 13:45:00 | 287.50 | 286.91 | 286.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 15:00:00 | 289.80 | 287.49 | 286.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 09:15:00 | 296.40 | 301.21 | 301.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 09:15:00 | 296.40 | 301.21 | 301.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 15:15:00 | 294.95 | 295.74 | 297.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 296.15 | 295.82 | 297.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 296.15 | 295.82 | 297.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 296.15 | 295.82 | 297.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 12:30:00 | 293.40 | 295.83 | 296.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 15:15:00 | 293.45 | 295.56 | 296.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 10:45:00 | 293.60 | 294.69 | 295.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 11:45:00 | 293.40 | 294.44 | 295.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 296.10 | 294.50 | 295.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 298.60 | 294.50 | 295.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 297.55 | 295.11 | 295.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:00:00 | 297.55 | 295.11 | 295.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-12 11:15:00 | 298.75 | 295.84 | 295.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 298.75 | 295.84 | 295.61 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 10:15:00 | 293.80 | 295.49 | 295.62 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 12:15:00 | 297.15 | 295.78 | 295.72 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 295.15 | 295.61 | 295.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 12:15:00 | 296.50 | 295.68 | 295.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 14:15:00 | 296.85 | 295.96 | 295.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 303.30 | 303.76 | 301.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 15:00:00 | 303.30 | 303.76 | 301.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 299.65 | 302.94 | 301.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 301.45 | 302.94 | 301.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 302.00 | 302.75 | 301.33 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 14:15:00 | 297.65 | 300.53 | 300.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 15:15:00 | 295.20 | 299.47 | 300.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 301.25 | 299.82 | 300.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 301.25 | 299.82 | 300.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 301.25 | 299.82 | 300.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 301.25 | 299.82 | 300.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 299.45 | 299.75 | 300.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 299.00 | 299.75 | 300.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 299.45 | 299.47 | 299.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:15:00 | 297.60 | 299.40 | 299.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 10:00:00 | 297.70 | 297.51 | 298.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 14:30:00 | 296.80 | 297.82 | 298.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:45:00 | 297.70 | 298.07 | 298.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 298.95 | 298.16 | 298.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 299.60 | 298.16 | 298.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 298.50 | 298.23 | 298.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-27 11:15:00 | 298.55 | 298.29 | 298.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 298.55 | 298.29 | 298.27 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 12:15:00 | 295.60 | 297.75 | 298.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 15:15:00 | 294.55 | 296.58 | 297.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 290.00 | 288.75 | 291.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 290.00 | 288.75 | 291.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 290.00 | 288.75 | 291.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:30:00 | 291.80 | 288.75 | 291.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 292.55 | 289.57 | 290.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:00:00 | 292.55 | 289.57 | 290.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 292.15 | 290.09 | 290.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:30:00 | 293.40 | 290.09 | 290.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 292.90 | 290.94 | 291.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 11:00:00 | 290.20 | 290.79 | 291.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 10:15:00 | 291.00 | 287.28 | 287.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 10:45:00 | 290.00 | 288.07 | 288.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 11:15:00 | 290.35 | 288.07 | 288.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 289.00 | 288.25 | 288.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 11:15:00 | 289.00 | 288.25 | 288.17 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 287.00 | 288.00 | 288.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 13:15:00 | 285.95 | 287.59 | 287.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 15:15:00 | 287.50 | 287.42 | 287.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 09:15:00 | 287.65 | 287.42 | 287.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 289.25 | 287.79 | 287.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:00:00 | 289.25 | 287.79 | 287.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 10:15:00 | 289.65 | 288.16 | 288.04 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 286.00 | 287.73 | 287.85 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 13:15:00 | 288.95 | 287.98 | 287.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 294.45 | 289.42 | 288.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 10:15:00 | 294.75 | 295.04 | 292.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 11:00:00 | 294.75 | 295.04 | 292.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 294.00 | 295.56 | 294.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 294.00 | 295.56 | 294.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 294.80 | 295.41 | 294.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:30:00 | 293.70 | 295.41 | 294.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 15:15:00 | 291.95 | 294.72 | 294.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 295.90 | 294.72 | 294.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 10:00:00 | 295.75 | 294.92 | 294.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 09:15:00 | 299.25 | 295.60 | 295.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 12:15:00 | 294.40 | 296.17 | 296.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 294.40 | 296.17 | 296.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 11:15:00 | 293.25 | 294.63 | 295.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 12:15:00 | 295.05 | 294.72 | 295.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 12:15:00 | 295.05 | 294.72 | 295.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 295.05 | 294.72 | 295.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:30:00 | 294.95 | 294.72 | 295.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 293.90 | 294.55 | 295.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 10:15:00 | 292.70 | 294.33 | 294.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:45:00 | 291.70 | 293.72 | 294.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 10:15:00 | 291.70 | 290.26 | 290.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 12:15:00 | 290.80 | 290.37 | 290.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 12:15:00 | 290.80 | 290.37 | 290.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 292.35 | 290.86 | 290.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 10:15:00 | 290.90 | 291.25 | 290.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 10:15:00 | 290.90 | 291.25 | 290.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 290.90 | 291.25 | 290.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 290.90 | 291.25 | 290.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 289.45 | 290.89 | 290.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:45:00 | 290.00 | 290.89 | 290.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 12:15:00 | 287.75 | 290.26 | 290.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 14:15:00 | 284.60 | 288.67 | 289.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 09:15:00 | 285.65 | 285.57 | 287.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 10:00:00 | 285.65 | 285.57 | 287.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 286.35 | 285.72 | 287.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:30:00 | 286.20 | 285.72 | 287.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 285.60 | 285.70 | 286.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 13:30:00 | 284.70 | 285.77 | 286.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 09:15:00 | 288.95 | 286.67 | 286.91 | SL hit (close>static) qty=1.00 sl=287.55 alert=retest2 |

### Cycle 22 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 290.70 | 287.48 | 287.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 291.40 | 288.74 | 287.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 15:15:00 | 288.20 | 289.44 | 288.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 15:15:00 | 288.20 | 289.44 | 288.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 15:15:00 | 288.20 | 289.44 | 288.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 09:45:00 | 290.30 | 289.50 | 288.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 15:15:00 | 286.00 | 288.55 | 288.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 286.00 | 288.55 | 288.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 279.55 | 286.75 | 287.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 15:15:00 | 278.00 | 275.45 | 278.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 15:15:00 | 278.00 | 275.45 | 278.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 278.00 | 275.45 | 278.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:15:00 | 278.90 | 275.45 | 278.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 280.10 | 276.38 | 278.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:00:00 | 280.10 | 276.38 | 278.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 281.70 | 277.45 | 279.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:30:00 | 281.45 | 277.45 | 279.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 14:15:00 | 281.40 | 280.08 | 280.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 286.55 | 281.68 | 280.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 15:15:00 | 281.35 | 283.29 | 282.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 15:15:00 | 281.35 | 283.29 | 282.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 281.35 | 283.29 | 282.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:30:00 | 281.45 | 282.73 | 282.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 279.15 | 282.02 | 281.78 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 279.20 | 281.45 | 281.54 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 282.35 | 281.59 | 281.54 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 279.25 | 281.19 | 281.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 15:15:00 | 278.60 | 280.45 | 281.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 15:15:00 | 279.20 | 278.52 | 279.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 09:15:00 | 274.40 | 278.52 | 279.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 274.80 | 277.78 | 279.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 15:15:00 | 272.90 | 275.10 | 277.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:45:00 | 272.75 | 271.53 | 272.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 12:15:00 | 272.90 | 271.90 | 272.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 13:15:00 | 273.05 | 272.17 | 272.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 14:15:00 | 275.40 | 273.01 | 272.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 275.40 | 273.01 | 272.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 09:15:00 | 276.00 | 274.02 | 273.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 15:15:00 | 275.00 | 275.25 | 274.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 13:15:00 | 275.45 | 275.68 | 275.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 275.45 | 275.68 | 275.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 275.45 | 275.68 | 275.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 278.95 | 278.64 | 277.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:45:00 | 278.75 | 278.64 | 277.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 277.45 | 278.46 | 277.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:30:00 | 277.05 | 278.46 | 277.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 277.15 | 278.20 | 277.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:30:00 | 276.65 | 278.20 | 277.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 277.30 | 277.54 | 277.56 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 279.05 | 277.77 | 277.66 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 10:15:00 | 272.55 | 276.73 | 277.20 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 277.00 | 275.99 | 275.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 11:15:00 | 277.70 | 276.53 | 276.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 10:15:00 | 277.00 | 277.37 | 276.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 10:15:00 | 277.00 | 277.37 | 276.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 277.00 | 277.37 | 276.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:45:00 | 277.90 | 277.37 | 276.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 277.50 | 277.40 | 276.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:30:00 | 276.45 | 277.40 | 276.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 276.30 | 277.18 | 276.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:45:00 | 276.40 | 277.18 | 276.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 276.15 | 276.97 | 276.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 276.15 | 276.97 | 276.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 276.70 | 277.94 | 277.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:00:00 | 276.70 | 277.94 | 277.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 277.30 | 277.81 | 277.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:30:00 | 276.05 | 277.81 | 277.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 276.80 | 277.61 | 277.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 280.95 | 277.61 | 277.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 13:15:00 | 281.25 | 283.77 | 284.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 281.25 | 283.77 | 284.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 13:15:00 | 280.00 | 281.08 | 281.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-18 09:15:00 | 281.35 | 280.80 | 281.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 281.35 | 280.80 | 281.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 281.35 | 280.80 | 281.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:45:00 | 282.00 | 280.80 | 281.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 280.70 | 280.78 | 281.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:45:00 | 282.25 | 280.78 | 281.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 284.30 | 281.49 | 281.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:00:00 | 284.30 | 281.49 | 281.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 283.00 | 281.79 | 281.68 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 280.35 | 281.40 | 281.54 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 12:15:00 | 284.20 | 281.96 | 281.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 13:15:00 | 285.80 | 282.73 | 282.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 14:15:00 | 287.00 | 288.05 | 285.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 15:00:00 | 287.00 | 288.05 | 285.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 288.50 | 288.14 | 286.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 285.60 | 287.85 | 286.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 287.35 | 287.75 | 286.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 11:30:00 | 288.75 | 288.20 | 286.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 13:00:00 | 288.00 | 288.16 | 286.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 15:15:00 | 289.00 | 287.44 | 286.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 11:45:00 | 287.95 | 287.51 | 286.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 285.90 | 287.19 | 286.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 285.90 | 287.19 | 286.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 286.15 | 286.98 | 286.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:45:00 | 285.80 | 286.98 | 286.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 285.00 | 286.58 | 286.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 14:15:00 | 285.00 | 286.58 | 286.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 15:15:00 | 284.30 | 286.13 | 286.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 14:15:00 | 284.85 | 284.54 | 285.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 284.85 | 284.54 | 285.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 286.45 | 284.77 | 285.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 286.25 | 284.77 | 285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 286.50 | 285.12 | 285.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:00:00 | 286.50 | 285.12 | 285.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 287.35 | 285.86 | 285.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 287.65 | 286.40 | 285.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 15:15:00 | 286.40 | 286.40 | 286.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 10:15:00 | 286.45 | 286.65 | 286.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 286.45 | 286.65 | 286.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:15:00 | 284.90 | 286.65 | 286.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 282.50 | 285.82 | 285.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 281.70 | 284.39 | 285.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 284.10 | 283.84 | 284.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 09:45:00 | 283.50 | 283.84 | 284.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 280.30 | 283.13 | 284.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 11:30:00 | 280.00 | 282.60 | 283.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 288.25 | 283.84 | 284.05 | SL hit (close>static) qty=1.00 sl=285.20 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 287.25 | 284.52 | 284.34 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 283.50 | 284.29 | 284.34 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 286.00 | 284.44 | 284.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 10:15:00 | 287.30 | 285.01 | 284.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 09:15:00 | 286.00 | 286.53 | 285.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 286.00 | 286.53 | 285.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 286.00 | 286.53 | 285.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:00:00 | 286.00 | 286.53 | 285.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 286.20 | 286.55 | 285.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:45:00 | 286.20 | 286.55 | 285.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 285.50 | 286.34 | 285.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:00:00 | 285.50 | 286.34 | 285.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 288.00 | 286.67 | 286.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:30:00 | 286.40 | 286.67 | 286.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 290.45 | 288.29 | 287.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 292.25 | 287.97 | 287.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 287.35 | 287.46 | 287.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 287.35 | 287.46 | 287.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 10:15:00 | 287.25 | 287.42 | 287.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 11:15:00 | 287.65 | 287.46 | 287.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 11:15:00 | 287.65 | 287.46 | 287.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 287.65 | 287.46 | 287.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:00:00 | 287.65 | 287.46 | 287.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 287.40 | 287.45 | 287.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 09:30:00 | 286.15 | 287.27 | 287.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 13:30:00 | 286.80 | 287.30 | 287.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 14:15:00 | 288.90 | 287.62 | 287.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 14:15:00 | 288.90 | 287.62 | 287.51 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 12:15:00 | 285.00 | 287.01 | 287.26 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 11:15:00 | 297.70 | 288.37 | 287.61 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 287.20 | 288.06 | 288.06 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 15:15:00 | 289.40 | 288.03 | 287.98 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 287.35 | 287.89 | 287.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 12:15:00 | 286.00 | 287.36 | 287.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 288.65 | 286.96 | 287.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 288.65 | 286.96 | 287.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 288.65 | 286.96 | 287.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:30:00 | 284.95 | 286.17 | 286.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:45:00 | 283.80 | 285.82 | 286.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 270.70 | 279.86 | 283.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 269.61 | 279.86 | 283.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-25 13:15:00 | 277.00 | 276.89 | 279.87 | SL hit (close>ema200) qty=0.50 sl=276.89 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 15:15:00 | 266.50 | 265.86 | 265.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 268.45 | 266.38 | 266.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 275.00 | 277.06 | 274.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 10:45:00 | 274.95 | 277.06 | 274.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 273.95 | 276.09 | 274.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 15:00:00 | 275.00 | 275.47 | 274.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 13:15:00 | 275.20 | 274.77 | 274.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 13:15:00 | 271.80 | 274.17 | 274.12 | SL hit (close<static) qty=1.00 sl=273.35 alert=retest2 |

### Cycle 51 — SELL (started 2023-11-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 14:15:00 | 270.55 | 273.45 | 273.80 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 12:15:00 | 274.45 | 273.89 | 273.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 277.25 | 274.77 | 274.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 11:15:00 | 279.10 | 279.63 | 278.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 11:15:00 | 279.10 | 279.63 | 278.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 279.10 | 279.63 | 278.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 12:00:00 | 279.10 | 279.63 | 278.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 277.45 | 279.09 | 278.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:00:00 | 277.45 | 279.09 | 278.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 275.50 | 278.37 | 278.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 15:00:00 | 275.50 | 278.37 | 278.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 15:15:00 | 274.75 | 277.65 | 277.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 10:15:00 | 271.65 | 275.16 | 275.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 273.65 | 273.32 | 274.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 273.65 | 273.32 | 274.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 273.65 | 273.32 | 274.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:00:00 | 273.65 | 273.32 | 274.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 272.95 | 273.15 | 273.99 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 276.15 | 274.50 | 274.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 09:15:00 | 278.20 | 275.98 | 275.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 12:15:00 | 275.45 | 276.07 | 275.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 12:15:00 | 275.45 | 276.07 | 275.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 275.45 | 276.07 | 275.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:00:00 | 275.45 | 276.07 | 275.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 275.10 | 275.88 | 275.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:45:00 | 275.15 | 275.88 | 275.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 277.90 | 276.28 | 275.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 09:15:00 | 282.15 | 276.63 | 275.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 14:15:00 | 278.50 | 278.22 | 277.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 14:45:00 | 281.65 | 279.16 | 277.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 12:15:00 | 283.80 | 287.28 | 287.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 283.80 | 287.28 | 287.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 11:15:00 | 283.40 | 285.48 | 286.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 15:15:00 | 283.70 | 283.68 | 285.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-07 09:15:00 | 281.75 | 283.68 | 285.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 286.80 | 284.31 | 285.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:45:00 | 287.25 | 284.31 | 285.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 288.55 | 285.15 | 285.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:45:00 | 288.35 | 285.15 | 285.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 287.05 | 285.95 | 285.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 288.30 | 286.67 | 286.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 10:15:00 | 296.50 | 297.95 | 295.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 11:00:00 | 296.50 | 297.95 | 295.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 295.20 | 297.40 | 295.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:00:00 | 295.20 | 297.40 | 295.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 296.55 | 297.23 | 295.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:45:00 | 295.40 | 297.23 | 295.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 294.85 | 296.75 | 295.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:00:00 | 294.85 | 296.75 | 295.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 293.80 | 296.16 | 295.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 294.20 | 296.16 | 295.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 295.40 | 295.44 | 295.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 297.70 | 295.61 | 295.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 09:30:00 | 297.70 | 295.81 | 295.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 12:15:00 | 294.10 | 295.09 | 295.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 12:15:00 | 294.10 | 295.09 | 295.14 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 298.15 | 295.09 | 295.05 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 10:15:00 | 292.85 | 294.64 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 11:15:00 | 292.55 | 294.22 | 294.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 09:15:00 | 291.45 | 291.31 | 292.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 291.45 | 291.31 | 292.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 291.45 | 291.31 | 292.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:00:00 | 291.45 | 291.31 | 292.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 287.85 | 287.83 | 289.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 287.85 | 287.83 | 289.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 290.55 | 288.32 | 289.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:45:00 | 291.95 | 288.32 | 289.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 291.85 | 289.03 | 289.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 291.85 | 289.03 | 289.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 12:15:00 | 290.85 | 289.84 | 289.73 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 282.70 | 288.41 | 289.09 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 290.00 | 288.60 | 288.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 294.50 | 289.78 | 289.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 13:15:00 | 291.00 | 291.21 | 290.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 13:15:00 | 291.00 | 291.21 | 290.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 291.00 | 291.21 | 290.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:45:00 | 290.65 | 291.21 | 290.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 294.55 | 291.88 | 290.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:45:00 | 291.00 | 291.88 | 290.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 11:15:00 | 294.30 | 295.58 | 294.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 11:45:00 | 294.35 | 295.58 | 294.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 293.20 | 295.11 | 294.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 293.45 | 295.11 | 294.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 294.10 | 294.91 | 294.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 14:15:00 | 295.00 | 294.91 | 294.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 14:45:00 | 295.00 | 294.92 | 294.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-02 09:15:00 | 324.50 | 318.65 | 312.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 326.75 | 330.29 | 330.33 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 333.75 | 330.38 | 330.32 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 13:15:00 | 328.10 | 330.43 | 330.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 327.95 | 329.93 | 330.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 321.20 | 321.15 | 324.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 321.20 | 321.15 | 324.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 321.20 | 321.15 | 324.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:45:00 | 321.70 | 321.15 | 324.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 324.00 | 319.68 | 321.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 12:45:00 | 320.50 | 320.53 | 321.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 320.90 | 320.72 | 321.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 09:45:00 | 320.60 | 321.22 | 321.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 10:15:00 | 330.10 | 322.99 | 322.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 330.10 | 322.99 | 322.42 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 319.45 | 322.92 | 323.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 316.90 | 320.57 | 322.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 317.95 | 314.75 | 317.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 317.95 | 314.75 | 317.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 317.95 | 314.75 | 317.18 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 320.20 | 318.52 | 318.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 323.25 | 319.70 | 318.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 346.00 | 346.72 | 337.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 14:30:00 | 351.90 | 346.72 | 337.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 363.00 | 362.24 | 357.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 10:15:00 | 370.00 | 362.24 | 357.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 15:15:00 | 356.30 | 360.11 | 360.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 356.30 | 360.11 | 360.27 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 364.00 | 360.88 | 360.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 369.20 | 362.55 | 361.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 365.15 | 365.52 | 363.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 09:30:00 | 365.00 | 365.52 | 363.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 365.20 | 365.46 | 363.82 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 358.15 | 362.82 | 363.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 354.95 | 361.25 | 362.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 358.55 | 357.49 | 360.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 358.55 | 357.49 | 360.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 358.55 | 357.49 | 360.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:30:00 | 361.40 | 357.49 | 360.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 363.50 | 358.69 | 360.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:30:00 | 365.00 | 358.69 | 360.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 364.00 | 359.75 | 360.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 11:30:00 | 362.60 | 359.75 | 360.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 13:15:00 | 373.00 | 363.24 | 362.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 09:15:00 | 379.60 | 368.40 | 364.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 11:15:00 | 376.15 | 378.10 | 373.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-07 12:00:00 | 376.15 | 378.10 | 373.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 374.20 | 377.32 | 373.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:30:00 | 374.25 | 377.32 | 373.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 380.15 | 377.89 | 374.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 14:30:00 | 384.00 | 379.10 | 375.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 10:00:00 | 386.75 | 392.17 | 386.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:15:00 | 384.10 | 387.36 | 384.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 14:15:00 | 381.55 | 385.20 | 384.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 378.20 | 383.77 | 383.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 379.80 | 383.77 | 383.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-12 09:15:00 | 366.80 | 380.38 | 382.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 366.80 | 380.38 | 382.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 362.50 | 375.22 | 379.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 14:15:00 | 354.05 | 349.09 | 355.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 15:00:00 | 354.05 | 349.09 | 355.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 352.70 | 349.81 | 354.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 365.00 | 349.81 | 354.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 367.90 | 353.43 | 356.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:15:00 | 369.65 | 353.43 | 356.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 368.00 | 356.34 | 357.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 371.80 | 356.34 | 357.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 378.30 | 360.73 | 359.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 10:15:00 | 388.95 | 373.00 | 366.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 373.20 | 377.49 | 372.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 373.20 | 377.49 | 372.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 373.20 | 377.49 | 372.40 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 09:15:00 | 365.15 | 371.20 | 371.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 11:15:00 | 361.70 | 367.87 | 369.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 369.15 | 365.91 | 367.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 369.15 | 365.91 | 367.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 369.15 | 365.91 | 367.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:45:00 | 371.80 | 365.91 | 367.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 367.95 | 366.32 | 367.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:30:00 | 368.80 | 366.32 | 367.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 372.00 | 367.46 | 368.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:30:00 | 370.50 | 367.46 | 368.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 12:15:00 | 374.60 | 368.89 | 368.69 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 367.45 | 369.31 | 369.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 15:15:00 | 365.90 | 368.24 | 368.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 11:15:00 | 367.85 | 366.88 | 367.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 11:15:00 | 367.85 | 366.88 | 367.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 367.85 | 366.88 | 367.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:45:00 | 367.70 | 366.88 | 367.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 367.50 | 367.00 | 367.94 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 15:15:00 | 374.90 | 369.55 | 368.96 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 366.80 | 368.94 | 369.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 363.80 | 367.91 | 368.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 13:15:00 | 364.00 | 362.58 | 364.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 13:15:00 | 364.00 | 362.58 | 364.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 364.00 | 362.58 | 364.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 14:00:00 | 364.00 | 362.58 | 364.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 361.75 | 362.42 | 364.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 14:30:00 | 363.10 | 362.42 | 364.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 366.80 | 362.42 | 363.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:30:00 | 366.55 | 362.42 | 363.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 365.70 | 363.08 | 363.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:30:00 | 365.85 | 363.08 | 363.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 371.90 | 365.61 | 364.77 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 10:15:00 | 360.00 | 363.91 | 364.11 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 364.95 | 364.24 | 364.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 366.20 | 364.63 | 364.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 09:15:00 | 365.05 | 365.26 | 364.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-02 09:45:00 | 365.05 | 365.26 | 364.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 364.00 | 365.45 | 365.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:30:00 | 364.35 | 365.45 | 365.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 364.95 | 365.35 | 365.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 13:30:00 | 370.60 | 367.67 | 366.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 360.75 | 367.63 | 368.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 360.75 | 367.63 | 368.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 359.70 | 362.05 | 363.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 343.25 | 342.58 | 348.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 343.25 | 342.58 | 348.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 345.75 | 344.18 | 347.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 347.00 | 344.18 | 347.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 347.00 | 344.74 | 347.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 350.00 | 344.74 | 347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 347.05 | 345.21 | 347.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 12:00:00 | 341.80 | 344.40 | 346.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 13:45:00 | 341.05 | 343.59 | 346.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:00:00 | 339.30 | 342.73 | 345.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 15:00:00 | 341.25 | 343.21 | 344.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 344.15 | 343.41 | 344.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 344.15 | 343.41 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 341.90 | 343.11 | 344.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 345.30 | 343.11 | 344.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 344.65 | 342.74 | 343.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:00:00 | 344.65 | 342.74 | 343.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 342.00 | 342.59 | 343.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 14:15:00 | 341.05 | 342.59 | 343.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 10:15:00 | 345.00 | 342.53 | 343.06 | SL hit (close>static) qty=1.00 sl=344.80 alert=retest2 |

### Cycle 84 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 344.45 | 343.41 | 343.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 348.15 | 344.67 | 343.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 14:15:00 | 345.30 | 346.07 | 345.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-21 15:15:00 | 345.95 | 346.07 | 345.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 345.95 | 346.04 | 345.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 11:30:00 | 348.05 | 346.60 | 345.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-08 09:15:00 | 382.86 | 374.41 | 371.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 375.85 | 381.57 | 381.98 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 13:15:00 | 382.90 | 381.60 | 381.54 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 379.85 | 381.25 | 381.39 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 388.95 | 382.48 | 381.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 10:15:00 | 393.80 | 384.74 | 382.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 12:15:00 | 389.35 | 392.38 | 389.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 12:15:00 | 389.35 | 392.38 | 389.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 389.35 | 392.38 | 389.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:00:00 | 389.35 | 392.38 | 389.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 387.55 | 391.41 | 389.02 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 380.00 | 387.50 | 387.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 09:15:00 | 302.15 | 365.46 | 376.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 10:15:00 | 295.25 | 294.99 | 301.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-26 10:30:00 | 295.45 | 294.99 | 301.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 298.15 | 295.24 | 298.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 298.30 | 295.24 | 298.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 301.25 | 296.44 | 299.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 301.25 | 296.44 | 299.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 299.20 | 296.99 | 299.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:45:00 | 299.80 | 296.99 | 299.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 299.45 | 297.48 | 299.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:45:00 | 300.00 | 297.48 | 299.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 298.10 | 297.61 | 299.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:45:00 | 296.80 | 297.81 | 298.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:00:00 | 296.00 | 296.20 | 296.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:45:00 | 297.15 | 296.39 | 296.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 14:30:00 | 297.30 | 296.71 | 296.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 296.00 | 296.56 | 296.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 297.25 | 296.56 | 296.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 294.00 | 296.05 | 296.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 14:30:00 | 292.50 | 294.03 | 295.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 15:15:00 | 294.00 | 292.92 | 292.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 15:15:00 | 294.00 | 292.92 | 292.83 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 09:15:00 | 289.85 | 292.31 | 292.56 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 292.70 | 291.17 | 291.02 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 290.00 | 291.09 | 291.15 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 295.00 | 291.54 | 291.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 296.60 | 294.26 | 292.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 298.25 | 298.53 | 296.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 298.25 | 298.53 | 296.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 298.25 | 298.53 | 296.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 297.95 | 298.53 | 296.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 297.75 | 298.37 | 296.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 296.45 | 298.37 | 296.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 297.35 | 298.17 | 296.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 296.90 | 298.17 | 296.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 296.90 | 297.91 | 296.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 296.90 | 297.91 | 296.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 296.95 | 297.72 | 296.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 296.95 | 297.72 | 296.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 297.75 | 297.73 | 296.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:30:00 | 297.15 | 297.73 | 296.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 298.05 | 297.87 | 297.17 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 296.60 | 297.05 | 297.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 295.80 | 296.60 | 296.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 293.25 | 292.45 | 294.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 10:15:00 | 293.25 | 292.45 | 294.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 293.25 | 292.45 | 294.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 293.25 | 292.45 | 294.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 292.10 | 292.38 | 293.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:30:00 | 293.05 | 292.38 | 293.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 293.40 | 292.03 | 293.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 289.50 | 290.74 | 291.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:45:00 | 289.40 | 290.47 | 291.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 287.80 | 288.92 | 290.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 288.90 | 288.96 | 289.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 289.85 | 289.14 | 289.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 289.85 | 289.14 | 289.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 290.50 | 289.41 | 289.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 295.65 | 289.41 | 289.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 297.90 | 291.11 | 290.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 297.90 | 291.11 | 290.61 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 277.10 | 289.08 | 290.45 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 284.35 | 282.33 | 282.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 12:15:00 | 291.30 | 284.70 | 283.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 14:15:00 | 304.75 | 305.22 | 302.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 15:00:00 | 304.75 | 305.22 | 302.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 304.60 | 304.58 | 303.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:45:00 | 303.35 | 304.58 | 303.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 304.20 | 304.47 | 303.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 301.40 | 304.47 | 303.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 303.75 | 304.33 | 303.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 302.90 | 304.33 | 303.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 302.30 | 303.92 | 303.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 302.90 | 303.92 | 303.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 304.20 | 303.98 | 303.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:00:00 | 307.25 | 304.37 | 303.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 304.70 | 305.58 | 305.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 305.25 | 305.53 | 305.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 301.60 | 305.13 | 305.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 301.60 | 305.13 | 305.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 300.90 | 303.70 | 304.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 13:15:00 | 297.05 | 296.90 | 298.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 14:00:00 | 297.05 | 296.90 | 298.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 301.90 | 297.86 | 298.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 301.65 | 297.86 | 298.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 303.50 | 298.99 | 299.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 302.90 | 298.99 | 299.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 303.00 | 299.79 | 299.61 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 15:15:00 | 296.45 | 299.03 | 299.37 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 301.45 | 299.64 | 299.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 305.20 | 300.75 | 300.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 301.25 | 302.52 | 301.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 301.25 | 302.52 | 301.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 301.25 | 302.52 | 301.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 301.25 | 302.52 | 301.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 303.40 | 302.70 | 301.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:30:00 | 304.30 | 302.59 | 301.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:45:00 | 304.45 | 304.41 | 303.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 314.90 | 320.67 | 321.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 314.90 | 320.67 | 321.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 313.20 | 316.34 | 318.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 317.80 | 316.63 | 318.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 317.80 | 316.63 | 318.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 317.80 | 316.63 | 318.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 318.15 | 316.63 | 318.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 317.30 | 316.76 | 318.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 317.80 | 316.76 | 318.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 317.95 | 316.35 | 317.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:15:00 | 318.35 | 316.35 | 317.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 319.95 | 317.07 | 317.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 319.80 | 317.07 | 317.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 319.65 | 317.59 | 317.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 319.65 | 317.59 | 317.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 318.05 | 317.63 | 317.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 318.05 | 317.63 | 317.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 318.50 | 317.81 | 317.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 318.50 | 317.81 | 317.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 321.00 | 318.45 | 318.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 324.15 | 320.31 | 319.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 344.40 | 345.34 | 341.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 12:00:00 | 344.40 | 345.34 | 341.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 341.90 | 344.96 | 341.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 341.90 | 344.96 | 341.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 339.60 | 343.89 | 341.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 339.60 | 343.89 | 341.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 339.40 | 342.99 | 341.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 343.00 | 343.06 | 341.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 12:45:00 | 344.30 | 343.05 | 341.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 338.85 | 340.88 | 341.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 338.85 | 340.88 | 341.15 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 14:15:00 | 343.30 | 341.29 | 341.23 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 340.20 | 341.18 | 341.20 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 343.15 | 340.19 | 340.02 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 338.55 | 339.79 | 339.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 10:15:00 | 336.85 | 338.45 | 339.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 11:15:00 | 337.05 | 335.81 | 337.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 11:15:00 | 337.05 | 335.81 | 337.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 337.05 | 335.81 | 337.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 337.05 | 335.81 | 337.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 336.05 | 335.86 | 336.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 337.85 | 335.86 | 336.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 336.50 | 335.99 | 336.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 337.50 | 335.99 | 336.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 337.20 | 336.00 | 336.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:30:00 | 331.90 | 335.22 | 336.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 325.45 | 322.01 | 321.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 325.45 | 322.01 | 321.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 326.65 | 323.70 | 322.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 333.00 | 333.12 | 330.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:15:00 | 336.50 | 333.12 | 330.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 410.65 | 411.53 | 399.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 402.35 | 411.53 | 399.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 455.00 | 434.46 | 418.14 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 447.20 | 450.62 | 450.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 446.00 | 449.70 | 450.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 437.70 | 437.03 | 442.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 437.00 | 437.03 | 442.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 431.15 | 434.31 | 437.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 429.70 | 433.57 | 436.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 430.50 | 433.06 | 436.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 429.35 | 432.06 | 435.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:30:00 | 427.60 | 431.12 | 433.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 408.21 | 413.65 | 418.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 408.97 | 413.65 | 418.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 407.88 | 413.65 | 418.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 406.22 | 413.65 | 418.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 403.85 | 399.84 | 404.07 | SL hit (close>ema200) qty=0.50 sl=399.84 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 404.90 | 402.89 | 402.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 408.30 | 404.25 | 403.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 404.00 | 404.86 | 403.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 404.00 | 404.86 | 403.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 405.30 | 404.95 | 404.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:30:00 | 406.00 | 404.67 | 404.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:00:00 | 406.45 | 404.67 | 404.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 406.80 | 405.42 | 404.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 411.15 | 424.84 | 425.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 411.15 | 424.84 | 425.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 406.90 | 410.34 | 411.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 15:15:00 | 407.40 | 406.47 | 408.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 09:15:00 | 408.35 | 406.47 | 408.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 407.35 | 406.65 | 408.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 409.75 | 406.65 | 408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 408.75 | 407.07 | 408.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 408.75 | 407.07 | 408.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 407.85 | 407.22 | 408.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:30:00 | 410.50 | 407.22 | 408.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 406.50 | 407.08 | 408.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:45:00 | 405.50 | 406.51 | 407.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 405.00 | 405.21 | 406.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 404.70 | 405.96 | 406.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 385.22 | 397.26 | 401.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 384.75 | 397.26 | 401.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 384.46 | 397.26 | 401.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 400.35 | 396.87 | 399.39 | SL hit (close>ema200) qty=0.50 sl=396.87 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 393.20 | 389.39 | 389.25 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 387.55 | 389.04 | 389.11 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 390.25 | 389.28 | 389.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 393.60 | 390.26 | 389.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 390.35 | 390.91 | 390.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 12:15:00 | 390.35 | 390.91 | 390.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 390.35 | 390.91 | 390.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 390.35 | 390.91 | 390.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 390.00 | 390.73 | 390.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 390.00 | 390.73 | 390.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 388.50 | 390.28 | 390.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 388.50 | 390.28 | 390.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 387.45 | 389.72 | 389.77 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 390.95 | 389.72 | 389.71 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 385.15 | 388.85 | 389.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 384.15 | 386.57 | 387.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 388.55 | 384.96 | 386.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 388.55 | 384.96 | 386.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 388.55 | 384.96 | 386.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 388.55 | 384.96 | 386.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 384.70 | 384.91 | 385.91 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 394.75 | 387.73 | 386.96 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 380.85 | 387.32 | 387.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 379.40 | 382.15 | 384.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 382.55 | 382.23 | 384.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 10:45:00 | 382.60 | 382.23 | 384.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 382.65 | 382.12 | 383.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 384.45 | 382.12 | 383.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 381.70 | 382.03 | 383.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 379.50 | 381.08 | 382.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 360.52 | 368.50 | 374.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-18 09:15:00 | 341.55 | 350.98 | 359.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 340.65 | 334.86 | 334.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 344.40 | 338.69 | 336.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 341.75 | 342.98 | 340.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 341.75 | 342.98 | 340.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 341.75 | 342.98 | 340.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 341.45 | 342.98 | 340.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 342.55 | 342.63 | 340.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 341.10 | 342.63 | 340.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 341.75 | 342.33 | 340.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 341.20 | 342.33 | 340.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 349.45 | 346.33 | 344.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 352.55 | 349.30 | 346.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 352.90 | 351.29 | 348.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-10 14:15:00 | 387.81 | 378.55 | 376.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 380.00 | 381.43 | 381.43 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 15:15:00 | 382.85 | 381.71 | 381.56 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 372.30 | 379.83 | 380.72 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 383.25 | 379.66 | 379.45 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 374.05 | 378.70 | 379.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 372.65 | 377.49 | 378.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 370.05 | 368.88 | 371.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:30:00 | 370.05 | 368.88 | 371.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 370.05 | 369.11 | 371.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 370.05 | 369.11 | 371.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 371.00 | 369.49 | 371.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 371.00 | 369.49 | 371.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 371.25 | 369.84 | 371.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 371.25 | 369.84 | 371.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 370.55 | 369.98 | 371.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 369.40 | 369.98 | 371.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 373.45 | 370.50 | 371.21 | SL hit (close>static) qty=1.00 sl=372.90 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 361.80 | 359.10 | 358.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 362.90 | 359.86 | 359.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 367.05 | 368.54 | 366.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 367.05 | 368.54 | 366.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 367.50 | 368.33 | 366.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 363.95 | 368.33 | 366.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 362.35 | 367.14 | 365.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 362.35 | 367.14 | 365.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 359.20 | 365.55 | 365.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 359.20 | 365.55 | 365.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 361.95 | 364.83 | 365.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 358.55 | 362.84 | 364.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 356.20 | 354.59 | 357.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 356.20 | 354.59 | 357.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 356.20 | 354.59 | 357.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 345.60 | 352.53 | 354.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 11:30:00 | 348.65 | 350.80 | 353.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:15:00 | 348.85 | 350.80 | 353.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:45:00 | 348.65 | 350.27 | 352.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 346.00 | 342.26 | 345.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 349.30 | 347.06 | 346.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 349.30 | 347.06 | 346.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 382.85 | 357.30 | 352.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 362.85 | 364.70 | 358.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 362.85 | 364.70 | 358.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 362.85 | 365.79 | 363.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 362.85 | 365.79 | 363.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 360.85 | 364.80 | 362.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:45:00 | 360.10 | 364.80 | 362.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 362.80 | 364.29 | 363.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 362.80 | 364.29 | 363.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 365.80 | 364.59 | 363.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 370.55 | 364.59 | 363.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:45:00 | 367.90 | 364.92 | 363.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 11:15:00 | 359.90 | 363.44 | 363.25 | SL hit (close<static) qty=1.00 sl=361.80 alert=retest2 |

### Cycle 131 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 361.00 | 362.95 | 363.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 356.75 | 360.55 | 361.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 14:15:00 | 359.25 | 359.06 | 360.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 15:00:00 | 359.25 | 359.06 | 360.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 355.50 | 358.34 | 359.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 351.80 | 355.74 | 357.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 346.70 | 342.87 | 342.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 346.70 | 342.87 | 342.53 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 338.90 | 342.46 | 342.67 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 342.90 | 340.75 | 340.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 345.55 | 341.71 | 341.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 15:15:00 | 342.50 | 342.66 | 341.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:15:00 | 342.35 | 342.66 | 341.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 340.15 | 342.16 | 341.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 339.15 | 342.16 | 341.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 338.15 | 341.36 | 341.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 335.75 | 340.24 | 340.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 15:15:00 | 334.00 | 333.43 | 335.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 15:15:00 | 334.00 | 333.43 | 335.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 334.00 | 333.43 | 335.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 328.55 | 333.43 | 335.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 312.12 | 319.81 | 326.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 10:15:00 | 295.69 | 308.40 | 316.86 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 10:15:00 | 287.30 | 286.13 | 286.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 11:15:00 | 290.00 | 286.91 | 286.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 285.65 | 287.23 | 286.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 285.65 | 287.23 | 286.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 285.65 | 287.23 | 286.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 285.05 | 287.23 | 286.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 286.80 | 287.14 | 286.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:15:00 | 285.85 | 287.14 | 286.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 285.85 | 286.88 | 286.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 279.40 | 286.88 | 286.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 280.55 | 285.62 | 286.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 276.80 | 282.99 | 284.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 279.70 | 279.57 | 281.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 14:45:00 | 279.60 | 279.57 | 281.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 281.00 | 280.08 | 281.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:15:00 | 279.85 | 280.08 | 281.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 265.86 | 271.58 | 275.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 268.35 | 268.31 | 272.15 | SL hit (close>ema200) qty=0.50 sl=268.31 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 276.70 | 272.61 | 272.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 281.70 | 274.43 | 273.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 288.60 | 290.46 | 286.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 288.60 | 290.46 | 286.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 288.65 | 289.04 | 287.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 285.95 | 289.04 | 287.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 285.15 | 288.26 | 286.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 285.15 | 288.26 | 286.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 282.45 | 287.10 | 286.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 282.45 | 287.10 | 286.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 283.00 | 285.87 | 286.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 279.80 | 284.66 | 285.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 290.10 | 279.82 | 281.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 290.10 | 279.82 | 281.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 290.10 | 279.82 | 281.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 290.10 | 279.82 | 281.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 289.60 | 281.78 | 282.04 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 289.40 | 283.30 | 282.71 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 274.20 | 281.14 | 281.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 272.85 | 277.36 | 279.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 276.35 | 274.74 | 277.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 276.35 | 274.74 | 277.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 276.35 | 274.74 | 277.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 274.70 | 274.89 | 276.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:45:00 | 275.10 | 274.42 | 275.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:45:00 | 275.15 | 274.53 | 275.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 15:15:00 | 275.00 | 274.87 | 275.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 275.00 | 274.90 | 275.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 278.45 | 274.90 | 275.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 276.50 | 275.22 | 275.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 278.00 | 275.78 | 275.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 278.00 | 275.78 | 275.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 282.00 | 277.84 | 276.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 285.00 | 285.31 | 282.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:45:00 | 285.00 | 285.31 | 282.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 288.20 | 291.17 | 288.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 288.20 | 291.17 | 288.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 288.35 | 290.61 | 288.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 286.55 | 290.61 | 288.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 288.90 | 289.72 | 288.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:15:00 | 287.50 | 289.72 | 288.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 287.50 | 289.28 | 288.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 290.00 | 289.28 | 288.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 290.70 | 289.77 | 289.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 287.60 | 288.82 | 288.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 287.60 | 288.82 | 288.82 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 292.20 | 289.39 | 289.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 297.70 | 291.42 | 290.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 11:15:00 | 292.95 | 293.32 | 291.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 12:00:00 | 292.95 | 293.32 | 291.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 290.05 | 292.67 | 291.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 290.05 | 292.67 | 291.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 290.80 | 292.30 | 291.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 292.55 | 292.19 | 291.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 10:00:00 | 292.70 | 292.19 | 291.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 10:30:00 | 292.50 | 292.03 | 291.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 292.70 | 292.16 | 291.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 292.50 | 292.34 | 291.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 294.00 | 292.67 | 292.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 290.25 | 292.19 | 291.91 | SL hit (close<static) qty=1.00 sl=291.50 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 288.60 | 292.86 | 293.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 276.95 | 288.22 | 290.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 285.60 | 284.47 | 287.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 285.60 | 284.47 | 287.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 288.35 | 285.25 | 287.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 288.10 | 285.25 | 287.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 289.00 | 286.00 | 287.69 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 291.15 | 288.97 | 288.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 293.65 | 290.33 | 289.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 310.90 | 311.84 | 307.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:45:00 | 310.60 | 311.84 | 307.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 305.20 | 309.59 | 307.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 305.20 | 309.59 | 307.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 310.45 | 309.76 | 308.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 313.00 | 310.43 | 308.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:30:00 | 312.85 | 311.08 | 309.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:00:00 | 317.15 | 318.80 | 318.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 313.75 | 317.79 | 318.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 313.75 | 317.79 | 318.09 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 12:15:00 | 321.70 | 318.32 | 318.27 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 13:15:00 | 318.00 | 318.50 | 318.55 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 321.15 | 319.02 | 318.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 14:15:00 | 329.30 | 322.70 | 320.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 320.00 | 324.08 | 323.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 320.00 | 324.08 | 323.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 320.00 | 324.08 | 323.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 318.90 | 324.08 | 323.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 319.00 | 323.06 | 322.64 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 319.20 | 322.29 | 322.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 317.05 | 320.74 | 321.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 325.05 | 320.81 | 321.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 325.05 | 320.81 | 321.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 325.05 | 320.81 | 321.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 325.35 | 320.81 | 321.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 322.90 | 321.23 | 321.40 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 324.65 | 321.91 | 321.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 329.60 | 324.34 | 322.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 324.05 | 325.05 | 323.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 324.05 | 325.05 | 323.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 324.65 | 324.97 | 323.78 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 319.15 | 322.91 | 323.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 318.20 | 321.97 | 322.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 321.60 | 321.42 | 322.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 321.60 | 321.42 | 322.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 321.60 | 321.42 | 322.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 321.25 | 321.42 | 322.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 322.35 | 321.61 | 322.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 323.50 | 321.61 | 322.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 321.50 | 321.59 | 322.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 322.60 | 321.59 | 322.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 321.10 | 321.49 | 322.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 321.10 | 321.49 | 322.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 322.95 | 321.78 | 322.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 322.95 | 321.78 | 322.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 322.40 | 321.90 | 322.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 324.05 | 321.90 | 322.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 324.40 | 322.40 | 322.38 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 321.20 | 322.16 | 322.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 11:15:00 | 320.65 | 321.86 | 322.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 319.40 | 316.78 | 318.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 319.40 | 316.78 | 318.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 319.40 | 316.78 | 318.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 319.40 | 316.78 | 318.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 325.95 | 318.62 | 319.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 328.35 | 318.62 | 319.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 328.25 | 320.54 | 319.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 339.70 | 329.46 | 325.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 342.40 | 343.71 | 339.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 342.55 | 343.71 | 339.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 349.80 | 351.60 | 348.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:30:00 | 349.15 | 351.60 | 348.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 347.55 | 350.79 | 348.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 347.55 | 350.79 | 348.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 351.85 | 351.00 | 349.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:30:00 | 357.15 | 350.15 | 348.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 346.95 | 349.51 | 348.77 | SL hit (close<static) qty=1.00 sl=347.25 alert=retest2 |

### Cycle 157 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 343.35 | 348.28 | 348.28 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 349.25 | 347.87 | 347.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 350.65 | 348.64 | 348.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 344.80 | 348.25 | 348.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 344.80 | 348.25 | 348.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 344.80 | 348.25 | 348.05 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 344.30 | 347.46 | 347.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 341.65 | 345.66 | 346.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 346.55 | 345.40 | 346.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 346.55 | 345.40 | 346.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 346.55 | 345.40 | 346.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 346.55 | 345.40 | 346.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 347.60 | 345.84 | 346.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 339.55 | 345.84 | 346.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 332.80 | 328.92 | 328.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 332.80 | 328.92 | 328.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 334.80 | 332.16 | 330.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 333.50 | 334.07 | 332.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:45:00 | 333.65 | 334.07 | 332.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 331.70 | 333.60 | 332.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:45:00 | 331.60 | 333.60 | 332.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 330.15 | 332.91 | 332.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 330.15 | 332.91 | 332.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 332.00 | 332.73 | 332.21 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 329.75 | 331.82 | 331.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 329.15 | 330.46 | 331.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 332.75 | 330.78 | 331.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 332.75 | 330.78 | 331.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 332.75 | 330.78 | 331.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 332.75 | 330.78 | 331.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 332.60 | 331.14 | 331.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 332.65 | 331.14 | 331.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 333.85 | 331.68 | 331.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 10:15:00 | 336.50 | 333.77 | 332.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 335.00 | 335.71 | 334.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 335.00 | 335.71 | 334.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 335.15 | 335.60 | 334.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:30:00 | 336.25 | 335.84 | 334.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 336.15 | 336.09 | 335.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 330.50 | 334.49 | 334.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 330.50 | 334.49 | 334.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 330.05 | 332.78 | 333.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 329.70 | 329.44 | 330.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 329.70 | 329.44 | 330.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 330.20 | 329.56 | 330.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 330.20 | 329.56 | 330.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 330.00 | 329.65 | 330.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 329.25 | 329.65 | 330.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 327.90 | 329.40 | 330.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 312.79 | 319.89 | 324.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 318.30 | 317.92 | 321.95 | SL hit (close>ema200) qty=0.50 sl=317.92 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 322.35 | 318.79 | 318.48 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 318.00 | 318.85 | 318.87 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 13:15:00 | 319.55 | 318.99 | 318.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 328.00 | 321.00 | 319.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 328.35 | 329.30 | 325.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 328.35 | 329.30 | 325.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 326.00 | 328.16 | 326.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 326.00 | 328.16 | 326.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 325.90 | 327.71 | 326.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 325.75 | 327.71 | 326.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 329.50 | 328.07 | 326.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 334.75 | 329.68 | 327.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 334.30 | 330.60 | 328.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 334.60 | 331.47 | 329.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 337.35 | 332.66 | 330.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 334.70 | 337.08 | 335.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 334.80 | 337.08 | 335.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 334.85 | 336.63 | 335.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 335.40 | 335.71 | 335.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 334.05 | 335.38 | 335.27 | SL hit (close<static) qty=1.00 sl=334.20 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 333.95 | 335.09 | 335.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 333.30 | 334.74 | 334.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 333.75 | 333.61 | 334.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 333.75 | 333.61 | 334.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 333.75 | 333.61 | 334.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:45:00 | 335.60 | 333.61 | 334.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 333.25 | 333.17 | 333.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 334.35 | 333.17 | 333.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 330.00 | 330.60 | 331.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 328.25 | 331.41 | 331.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 327.85 | 330.77 | 331.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 328.80 | 327.87 | 329.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 331.15 | 326.47 | 325.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 331.15 | 326.47 | 325.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 334.70 | 329.65 | 327.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 333.15 | 333.54 | 331.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 333.15 | 333.54 | 331.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 333.35 | 333.27 | 331.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 332.55 | 333.27 | 331.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 328.05 | 333.27 | 332.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 328.10 | 333.27 | 332.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 327.35 | 332.09 | 331.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 327.35 | 332.09 | 331.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 325.75 | 330.82 | 331.30 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 335.45 | 331.54 | 331.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 342.20 | 334.53 | 332.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 334.85 | 335.16 | 333.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 334.85 | 335.16 | 333.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 334.85 | 335.16 | 333.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 334.70 | 335.16 | 333.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 330.85 | 334.30 | 333.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 330.85 | 334.30 | 333.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 330.15 | 333.47 | 332.97 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 331.40 | 332.59 | 332.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 327.35 | 331.08 | 331.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 320.85 | 320.67 | 323.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 320.85 | 320.67 | 323.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 323.00 | 320.95 | 322.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 323.00 | 320.95 | 322.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 323.00 | 321.36 | 322.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 322.00 | 321.36 | 322.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 319.50 | 322.53 | 322.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 305.90 | 310.04 | 314.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 303.52 | 310.04 | 314.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 307.60 | 307.34 | 310.83 | SL hit (close>ema200) qty=0.50 sl=307.34 alert=retest2 |

### Cycle 172 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 308.00 | 305.10 | 305.05 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 301.50 | 304.49 | 304.85 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 309.20 | 304.69 | 304.37 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 303.15 | 304.42 | 304.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 15:15:00 | 302.90 | 304.11 | 304.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 302.55 | 302.10 | 302.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 302.55 | 302.10 | 302.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 302.55 | 302.10 | 302.92 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 304.35 | 303.32 | 303.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 306.05 | 304.31 | 303.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 302.10 | 304.97 | 304.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 302.10 | 304.97 | 304.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 302.10 | 304.97 | 304.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 302.10 | 304.97 | 304.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 302.70 | 304.52 | 304.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 302.35 | 304.52 | 304.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 303.85 | 304.11 | 304.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 302.95 | 303.88 | 304.01 | Break + close below crossover candle low |

### Cycle 178 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 308.90 | 304.74 | 304.37 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 304.05 | 304.85 | 304.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 302.60 | 304.18 | 304.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 290.95 | 290.89 | 293.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 290.95 | 290.89 | 293.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 292.65 | 291.47 | 293.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 292.65 | 291.47 | 293.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 293.65 | 291.91 | 293.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 294.60 | 291.91 | 293.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 293.10 | 292.15 | 293.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 293.10 | 292.15 | 293.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 292.90 | 292.30 | 293.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 294.90 | 292.30 | 293.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 298.15 | 293.47 | 293.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 298.15 | 293.47 | 293.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 301.30 | 295.03 | 294.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 303.80 | 298.72 | 296.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 302.70 | 303.02 | 301.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 302.70 | 303.02 | 301.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 302.70 | 303.02 | 301.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 302.75 | 303.02 | 301.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 300.40 | 302.25 | 301.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 300.40 | 302.25 | 301.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 300.40 | 301.88 | 301.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 299.60 | 301.88 | 301.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 299.65 | 301.10 | 300.97 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 299.95 | 300.87 | 300.87 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 301.35 | 300.97 | 300.92 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 299.15 | 300.60 | 300.76 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 304.50 | 300.73 | 300.62 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 298.60 | 300.80 | 301.07 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 309.00 | 301.56 | 301.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 316.95 | 310.54 | 308.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 15:15:00 | 317.20 | 318.13 | 315.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:45:00 | 317.00 | 317.83 | 315.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 315.00 | 317.26 | 315.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 314.75 | 317.26 | 315.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 316.35 | 317.08 | 315.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 319.00 | 315.99 | 315.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 317.20 | 316.24 | 315.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:00:00 | 317.20 | 316.43 | 315.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:30:00 | 317.15 | 316.50 | 315.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 314.50 | 316.10 | 315.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 314.50 | 316.10 | 315.81 | SL hit (close<static) qty=1.00 sl=314.65 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 313.15 | 315.51 | 315.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 311.05 | 314.01 | 314.84 | Break + close below crossover candle low |

### Cycle 188 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 325.10 | 315.74 | 315.17 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 316.55 | 318.05 | 318.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 315.00 | 317.44 | 317.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 312.95 | 311.88 | 313.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 312.80 | 312.06 | 313.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 312.80 | 312.06 | 313.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 312.80 | 312.06 | 313.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 310.00 | 309.51 | 311.55 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 315.00 | 310.50 | 310.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 317.05 | 311.81 | 311.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 321.00 | 321.73 | 318.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 321.00 | 321.73 | 318.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 322.80 | 323.35 | 321.69 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 317.50 | 320.65 | 320.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 316.05 | 319.73 | 320.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 319.50 | 319.05 | 319.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 319.50 | 319.05 | 319.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 316.15 | 318.47 | 319.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 314.70 | 317.45 | 318.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 314.70 | 316.78 | 318.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 320.40 | 317.97 | 318.50 | SL hit (close>static) qty=1.00 sl=320.30 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 320.35 | 318.87 | 318.82 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 316.55 | 318.69 | 318.78 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 320.75 | 318.83 | 318.72 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 315.95 | 318.61 | 318.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 09:15:00 | 314.35 | 317.34 | 318.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 317.65 | 317.25 | 317.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 317.65 | 317.25 | 317.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 317.65 | 317.25 | 317.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 317.60 | 317.25 | 317.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 319.95 | 317.79 | 318.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 319.60 | 317.79 | 318.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 319.25 | 318.08 | 318.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 320.40 | 318.08 | 318.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 323.15 | 319.08 | 318.58 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 315.80 | 318.44 | 318.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 314.90 | 316.86 | 317.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 315.95 | 315.91 | 317.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 315.95 | 315.91 | 317.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 311.65 | 311.85 | 313.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:45:00 | 311.25 | 311.58 | 313.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:45:00 | 311.00 | 311.67 | 312.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 314.15 | 313.07 | 313.30 | SL hit (close>static) qty=1.00 sl=314.10 alert=retest2 |

### Cycle 198 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 315.10 | 313.48 | 313.46 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 311.55 | 313.51 | 313.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 311.10 | 312.67 | 313.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 313.50 | 310.55 | 311.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 313.50 | 310.55 | 311.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 313.50 | 310.55 | 311.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 313.50 | 310.55 | 311.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 316.30 | 311.70 | 311.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 316.80 | 311.70 | 311.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 318.50 | 313.06 | 312.38 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 311.55 | 312.50 | 312.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 310.50 | 311.37 | 311.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 311.05 | 310.27 | 310.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 311.05 | 310.27 | 310.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 311.05 | 310.27 | 310.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 311.05 | 310.27 | 310.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 309.95 | 310.21 | 310.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:45:00 | 309.55 | 310.22 | 310.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:30:00 | 309.10 | 309.89 | 310.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 301.85 | 300.63 | 300.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 301.85 | 300.63 | 300.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 305.00 | 302.43 | 301.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 301.05 | 302.16 | 301.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 301.05 | 302.16 | 301.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 301.05 | 302.16 | 301.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 301.05 | 302.16 | 301.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 304.10 | 302.55 | 301.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 304.80 | 302.94 | 302.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 13:15:00 | 299.70 | 302.17 | 302.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 299.70 | 302.17 | 302.22 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 303.05 | 300.78 | 300.60 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 297.55 | 300.12 | 300.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 295.00 | 297.53 | 298.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 11:15:00 | 294.65 | 294.53 | 295.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 11:45:00 | 294.60 | 294.53 | 295.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 294.70 | 294.59 | 295.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 294.50 | 294.59 | 295.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 294.50 | 294.57 | 295.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:00:00 | 294.10 | 294.57 | 295.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 294.00 | 294.56 | 295.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 307.05 | 293.46 | 292.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 307.05 | 293.46 | 292.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 313.85 | 297.54 | 294.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 298.05 | 301.17 | 297.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 298.05 | 301.17 | 297.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 295.40 | 300.01 | 297.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 299.20 | 300.01 | 297.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:30:00 | 299.30 | 299.17 | 297.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 300.00 | 299.17 | 297.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 296.35 | 297.72 | 297.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 296.35 | 297.72 | 297.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 294.20 | 296.81 | 297.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 284.15 | 283.29 | 284.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 284.15 | 283.29 | 284.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 284.15 | 283.29 | 284.95 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 288.55 | 286.11 | 285.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 289.40 | 286.77 | 286.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 283.80 | 286.55 | 286.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 283.80 | 286.55 | 286.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 283.80 | 286.55 | 286.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 283.80 | 286.55 | 286.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 286.55 | 286.55 | 286.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 287.40 | 286.55 | 286.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 14:15:00 | 287.65 | 286.56 | 286.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 283.90 | 286.05 | 286.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 283.90 | 286.05 | 286.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 282.50 | 285.34 | 285.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 280.50 | 280.33 | 281.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:30:00 | 280.60 | 280.33 | 281.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 280.55 | 280.37 | 281.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 281.15 | 280.37 | 281.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 288.20 | 282.21 | 282.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 288.20 | 282.21 | 282.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 288.30 | 283.43 | 282.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 289.65 | 285.28 | 283.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 295.55 | 296.40 | 293.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 295.55 | 296.40 | 293.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 294.55 | 296.05 | 294.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 294.55 | 296.05 | 294.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 297.50 | 296.34 | 294.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 295.60 | 296.34 | 294.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 296.20 | 296.31 | 294.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:45:00 | 296.85 | 296.22 | 294.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 296.85 | 296.22 | 294.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 297.50 | 296.62 | 295.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 293.30 | 295.68 | 295.44 | SL hit (close<static) qty=1.00 sl=293.80 alert=retest2 |

### Cycle 211 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 293.15 | 295.17 | 295.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 291.80 | 294.50 | 294.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 295.70 | 294.31 | 294.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 295.70 | 294.31 | 294.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 295.70 | 294.31 | 294.69 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 296.05 | 294.84 | 294.84 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 294.70 | 294.82 | 294.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 15:15:00 | 293.60 | 294.57 | 294.72 | Break + close below crossover candle low |

### Cycle 214 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 301.75 | 296.01 | 295.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 306.35 | 298.08 | 296.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 318.90 | 319.93 | 316.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 319.65 | 319.93 | 316.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 315.65 | 318.51 | 317.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 316.05 | 318.51 | 317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 313.60 | 317.53 | 317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 313.60 | 317.53 | 317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 314.75 | 316.38 | 316.58 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 317.95 | 316.71 | 316.66 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 313.50 | 316.11 | 316.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 311.65 | 314.69 | 315.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 300.40 | 300.27 | 303.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 11:45:00 | 300.50 | 300.27 | 303.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 305.55 | 301.78 | 303.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 305.55 | 301.78 | 303.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 306.80 | 302.79 | 303.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 308.75 | 302.79 | 303.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 309.00 | 305.02 | 304.57 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 301.25 | 304.44 | 304.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 300.50 | 303.13 | 303.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 303.00 | 302.78 | 303.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 15:15:00 | 303.00 | 302.78 | 303.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 303.00 | 302.78 | 303.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 303.15 | 302.78 | 303.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 301.65 | 302.55 | 303.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 301.70 | 302.55 | 303.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 300.15 | 301.90 | 302.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 302.55 | 301.90 | 302.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 304.00 | 298.32 | 299.39 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 301.35 | 299.97 | 299.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 302.55 | 300.81 | 300.35 | Break + close above crossover candle high |

### Cycle 221 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 296.10 | 299.87 | 299.96 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 299.95 | 297.61 | 297.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 305.65 | 300.07 | 298.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 13:15:00 | 303.60 | 303.93 | 302.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 303.60 | 303.93 | 302.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 304.90 | 304.13 | 302.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 303.35 | 304.13 | 302.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 301.30 | 303.32 | 302.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 297.65 | 303.32 | 302.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 302.25 | 303.10 | 302.35 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 299.50 | 302.06 | 302.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 295.25 | 300.37 | 301.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 299.65 | 297.93 | 299.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 299.65 | 297.93 | 299.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 299.65 | 297.93 | 299.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 299.65 | 297.93 | 299.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 300.00 | 298.35 | 299.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 301.15 | 298.35 | 299.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 301.60 | 299.00 | 299.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 300.40 | 299.00 | 299.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 301.55 | 300.03 | 300.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 301.55 | 300.03 | 300.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 15:15:00 | 303.00 | 300.81 | 300.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 299.05 | 300.46 | 300.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 299.05 | 300.46 | 300.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 299.05 | 300.46 | 300.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 299.05 | 300.46 | 300.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 298.50 | 300.07 | 300.11 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 302.40 | 300.00 | 299.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 305.50 | 302.83 | 301.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 310.15 | 311.75 | 309.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 15:00:00 | 315.00 | 313.47 | 311.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 309.40 | 312.95 | 311.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 309.40 | 312.95 | 311.50 | SL hit (close<ema400) qty=1.00 sl=311.50 alert=retest1 |

### Cycle 227 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 309.40 | 310.85 | 310.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 308.05 | 310.29 | 310.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 309.50 | 309.48 | 310.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 309.50 | 309.48 | 310.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 310.15 | 309.61 | 310.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 310.15 | 309.61 | 310.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 309.85 | 309.66 | 310.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:15:00 | 310.80 | 309.66 | 310.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 310.00 | 309.73 | 310.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 310.00 | 309.73 | 310.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 308.75 | 309.53 | 310.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 311.95 | 309.53 | 310.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 311.50 | 309.93 | 310.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 308.70 | 309.64 | 309.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 303.15 | 302.29 | 302.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 303.15 | 302.29 | 302.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 305.30 | 302.89 | 302.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 15:15:00 | 303.45 | 303.67 | 303.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 303.45 | 303.67 | 303.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 303.45 | 303.67 | 303.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 305.25 | 303.67 | 303.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 306.50 | 304.16 | 303.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 305.25 | 305.39 | 304.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 298.70 | 304.19 | 304.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 298.70 | 304.19 | 304.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 285.80 | 296.91 | 300.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 289.20 | 289.13 | 294.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 289.20 | 289.13 | 294.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 291.30 | 287.03 | 290.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 291.30 | 287.03 | 290.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 291.35 | 287.90 | 290.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 290.30 | 287.90 | 290.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 293.55 | 289.28 | 290.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 293.55 | 289.28 | 290.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 295.35 | 290.50 | 291.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 295.35 | 290.50 | 291.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 295.20 | 292.11 | 291.81 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 278.35 | 289.70 | 290.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 274.45 | 286.65 | 289.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 11:15:00 | 277.70 | 276.09 | 280.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 288.15 | 278.47 | 280.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 288.15 | 278.47 | 280.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 305.70 | 278.47 | 280.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 288.25 | 280.43 | 280.90 | EMA400 retest candle locked (from downside) |

### Cycle 232 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 290.35 | 282.41 | 281.76 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 279.50 | 283.43 | 283.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 273.30 | 281.40 | 282.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 270.30 | 269.73 | 273.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 271.10 | 269.73 | 273.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 267.50 | 269.63 | 272.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 262.30 | 267.12 | 270.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:30:00 | 262.50 | 260.29 | 264.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 249.19 | 252.58 | 256.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 249.38 | 252.58 | 256.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 12:15:00 | 236.07 | 245.15 | 251.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 238.75 | 234.94 | 234.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 240.00 | 236.95 | 236.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 239.51 | 240.07 | 238.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 239.51 | 240.07 | 238.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 239.51 | 240.07 | 238.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 239.51 | 240.07 | 238.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 240.19 | 240.09 | 238.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 241.31 | 240.42 | 239.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:15:00 | 241.30 | 240.33 | 239.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 241.38 | 240.87 | 240.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 14:00:00 | 241.44 | 240.98 | 240.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 241.34 | 241.05 | 240.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 240.31 | 241.05 | 240.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 236.15 | 240.30 | 240.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 236.15 | 240.30 | 240.09 | SL hit (close<static) qty=1.00 sl=238.77 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 238.08 | 239.86 | 239.91 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 243.63 | 240.13 | 239.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 251.51 | 244.72 | 242.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 278.52 | 278.55 | 273.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 09:45:00 | 277.08 | 278.55 | 273.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 280.12 | 282.71 | 280.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 277.70 | 282.71 | 280.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 278.35 | 281.84 | 280.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:45:00 | 278.90 | 281.84 | 280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 277.50 | 280.97 | 279.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 276.89 | 280.97 | 279.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 276.57 | 279.04 | 279.12 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 281.98 | 279.39 | 279.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 282.54 | 280.38 | 279.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 281.73 | 281.80 | 280.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 281.73 | 281.80 | 280.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 281.15 | 281.65 | 280.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 284.75 | 281.65 | 280.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 288.50 | 292.16 | 292.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 288.50 | 292.16 | 292.19 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 13:45:00 | 287.50 | 2023-06-05 09:15:00 | 296.40 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2023-05-24 15:00:00 | 289.80 | 2023-06-05 09:15:00 | 296.40 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2023-06-08 12:30:00 | 293.40 | 2023-06-12 11:15:00 | 298.75 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2023-06-08 15:15:00 | 293.45 | 2023-06-12 11:15:00 | 298.75 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2023-06-09 10:45:00 | 293.60 | 2023-06-12 11:15:00 | 298.75 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2023-06-09 11:45:00 | 293.40 | 2023-06-12 11:15:00 | 298.75 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2023-06-22 11:15:00 | 297.60 | 2023-06-27 11:15:00 | 298.55 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-06-23 10:00:00 | 297.70 | 2023-06-27 11:15:00 | 298.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2023-06-23 14:30:00 | 296.80 | 2023-06-27 11:15:00 | 298.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-06-26 11:45:00 | 297.70 | 2023-06-27 11:15:00 | 298.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2023-07-04 11:00:00 | 290.20 | 2023-07-07 11:15:00 | 289.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2023-07-07 10:15:00 | 291.00 | 2023-07-07 11:15:00 | 289.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2023-07-07 10:45:00 | 290.00 | 2023-07-07 11:15:00 | 289.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2023-07-07 11:15:00 | 290.35 | 2023-07-07 11:15:00 | 289.00 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2023-07-14 09:15:00 | 295.90 | 2023-07-18 12:15:00 | 294.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-07-14 10:00:00 | 295.75 | 2023-07-18 12:15:00 | 294.40 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-07-17 09:15:00 | 299.25 | 2023-07-18 12:15:00 | 294.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-07-20 10:15:00 | 292.70 | 2023-07-25 12:15:00 | 290.80 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2023-07-20 11:45:00 | 291.70 | 2023-07-25 12:15:00 | 290.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2023-07-25 10:15:00 | 291.70 | 2023-07-25 12:15:00 | 290.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2023-07-28 13:30:00 | 284.70 | 2023-07-31 09:15:00 | 288.95 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-08-02 09:45:00 | 290.30 | 2023-08-02 15:15:00 | 286.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-08-14 15:15:00 | 272.90 | 2023-08-18 14:15:00 | 275.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-18 10:45:00 | 272.75 | 2023-08-18 14:15:00 | 275.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-08-18 12:15:00 | 272.90 | 2023-08-18 14:15:00 | 275.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-18 13:15:00 | 273.05 | 2023-08-18 14:15:00 | 275.40 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-09-06 09:15:00 | 280.95 | 2023-09-12 13:15:00 | 281.25 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-09-22 11:30:00 | 288.75 | 2023-09-25 14:15:00 | 285.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-09-22 13:00:00 | 288.00 | 2023-09-25 14:15:00 | 285.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-09-22 15:15:00 | 289.00 | 2023-09-25 14:15:00 | 285.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-09-25 11:45:00 | 287.95 | 2023-09-25 14:15:00 | 285.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-09-29 11:30:00 | 280.00 | 2023-10-03 09:15:00 | 288.25 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2023-10-10 09:15:00 | 292.25 | 2023-10-11 09:15:00 | 287.35 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-10-12 09:30:00 | 286.15 | 2023-10-12 14:15:00 | 288.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-10-12 13:30:00 | 286.80 | 2023-10-12 14:15:00 | 288.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-10-20 13:30:00 | 284.95 | 2023-10-23 14:15:00 | 270.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:45:00 | 283.80 | 2023-10-23 14:15:00 | 269.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:30:00 | 284.95 | 2023-10-25 13:15:00 | 277.00 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2023-10-23 09:45:00 | 283.80 | 2023-10-25 13:15:00 | 277.00 | STOP_HIT | 0.50 | 2.40% |
| BUY | retest2 | 2023-11-08 15:00:00 | 275.00 | 2023-11-09 13:15:00 | 271.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-11-09 13:15:00 | 275.20 | 2023-11-09 13:15:00 | 271.80 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-11-28 09:15:00 | 282.15 | 2023-12-05 12:15:00 | 283.80 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2023-11-28 14:15:00 | 278.50 | 2023-12-05 12:15:00 | 283.80 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2023-11-28 14:45:00 | 281.65 | 2023-12-05 12:15:00 | 283.80 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2023-12-13 15:15:00 | 297.70 | 2023-12-14 12:15:00 | 294.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-12-14 09:30:00 | 297.70 | 2023-12-14 12:15:00 | 294.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-12-27 14:15:00 | 295.00 | 2024-01-02 09:15:00 | 324.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-27 14:45:00 | 295.00 | 2024-01-02 09:15:00 | 324.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-12 12:45:00 | 320.50 | 2024-01-15 10:15:00 | 330.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-01-12 14:30:00 | 320.90 | 2024-01-15 10:15:00 | 330.10 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-01-15 09:45:00 | 320.60 | 2024-01-15 10:15:00 | 330.10 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-01-29 10:15:00 | 370.00 | 2024-01-30 15:15:00 | 356.30 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-02-07 14:30:00 | 384.00 | 2024-02-12 09:15:00 | 366.80 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-02-09 10:00:00 | 386.75 | 2024-02-12 09:15:00 | 366.80 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2024-02-09 12:15:00 | 384.10 | 2024-02-12 09:15:00 | 366.80 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2024-02-09 14:15:00 | 381.55 | 2024-02-12 09:15:00 | 366.80 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-03-04 13:30:00 | 370.60 | 2024-03-06 11:15:00 | 360.75 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-03-15 12:00:00 | 341.80 | 2024-03-20 10:15:00 | 345.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-03-15 13:45:00 | 341.05 | 2024-03-20 13:15:00 | 344.45 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-03-15 15:00:00 | 339.30 | 2024-03-20 13:15:00 | 344.45 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-03-18 15:00:00 | 341.25 | 2024-03-20 13:15:00 | 344.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-03-19 14:15:00 | 341.05 | 2024-03-20 13:15:00 | 344.45 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-03-22 11:30:00 | 348.05 | 2024-04-08 09:15:00 | 382.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-30 14:45:00 | 296.80 | 2024-05-08 15:15:00 | 294.00 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2024-05-03 12:00:00 | 296.00 | 2024-05-08 15:15:00 | 294.00 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2024-05-03 12:45:00 | 297.15 | 2024-05-08 15:15:00 | 294.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2024-05-03 14:30:00 | 297.30 | 2024-05-08 15:15:00 | 294.00 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2024-05-06 14:30:00 | 292.50 | 2024-05-08 15:15:00 | 294.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-05-30 09:45:00 | 289.50 | 2024-06-03 09:15:00 | 297.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-05-30 10:45:00 | 289.40 | 2024-06-03 09:15:00 | 297.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-05-31 09:30:00 | 287.80 | 2024-06-03 09:15:00 | 297.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-05-31 14:15:00 | 288.90 | 2024-06-03 09:15:00 | 297.90 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-06-20 10:00:00 | 307.25 | 2024-06-24 14:15:00 | 301.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-06-24 09:15:00 | 304.70 | 2024-06-24 14:15:00 | 301.60 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-06-24 09:45:00 | 305.25 | 2024-06-24 14:15:00 | 301.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-07-03 10:30:00 | 304.30 | 2024-07-19 09:15:00 | 314.90 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2024-07-04 10:45:00 | 304.45 | 2024-07-19 09:15:00 | 314.90 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2024-08-01 09:30:00 | 343.00 | 2024-08-02 10:15:00 | 338.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-01 12:45:00 | 344.30 | 2024-08-02 10:15:00 | 338.85 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-08-09 13:30:00 | 331.90 | 2024-08-20 10:15:00 | 325.45 | STOP_HIT | 1.00 | 1.94% |
| SELL | retest2 | 2024-09-11 11:45:00 | 429.70 | 2024-09-17 11:15:00 | 408.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 12:45:00 | 430.50 | 2024-09-17 11:15:00 | 408.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 13:30:00 | 429.35 | 2024-09-17 11:15:00 | 407.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 13:30:00 | 427.60 | 2024-09-17 11:15:00 | 406.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 11:45:00 | 429.70 | 2024-09-19 14:15:00 | 403.85 | STOP_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2024-09-11 12:45:00 | 430.50 | 2024-09-19 14:15:00 | 403.85 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2024-09-11 13:30:00 | 429.35 | 2024-09-19 14:15:00 | 403.85 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2024-09-12 13:30:00 | 427.60 | 2024-09-19 14:15:00 | 403.85 | STOP_HIT | 0.50 | 5.55% |
| BUY | retest2 | 2024-09-26 10:30:00 | 406.00 | 2024-10-07 09:15:00 | 411.15 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-09-26 11:00:00 | 406.45 | 2024-10-07 09:15:00 | 411.15 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2024-09-26 14:30:00 | 406.80 | 2024-10-07 09:15:00 | 411.15 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2024-10-15 13:45:00 | 405.50 | 2024-10-18 09:15:00 | 385.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:30:00 | 405.00 | 2024-10-18 09:15:00 | 384.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:45:00 | 404.70 | 2024-10-18 09:15:00 | 384.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:45:00 | 405.50 | 2024-10-18 14:15:00 | 400.35 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2024-10-16 09:30:00 | 405.00 | 2024-10-18 14:15:00 | 400.35 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2024-10-17 09:45:00 | 404.70 | 2024-10-18 14:15:00 | 400.35 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-11-12 11:45:00 | 379.50 | 2024-11-13 12:15:00 | 360.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 379.50 | 2024-11-18 09:15:00 | 341.55 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-29 13:30:00 | 352.55 | 2024-12-10 14:15:00 | 387.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:30:00 | 352.90 | 2024-12-10 14:15:00 | 388.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 09:15:00 | 369.40 | 2024-12-20 10:15:00 | 373.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-12-20 14:00:00 | 370.05 | 2024-12-30 09:15:00 | 351.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:00:00 | 370.05 | 2024-12-30 09:15:00 | 362.60 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2024-12-20 15:00:00 | 366.35 | 2025-01-01 10:15:00 | 361.80 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-01-10 09:15:00 | 345.60 | 2025-01-15 09:15:00 | 349.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-01-10 11:30:00 | 348.65 | 2025-01-15 09:15:00 | 349.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-01-10 12:15:00 | 348.85 | 2025-01-15 09:15:00 | 349.30 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-01-10 12:45:00 | 348.65 | 2025-01-15 09:15:00 | 349.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-01-22 09:15:00 | 370.55 | 2025-01-22 11:15:00 | 359.90 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-01-22 09:45:00 | 367.90 | 2025-01-22 11:15:00 | 359.90 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-01-24 14:30:00 | 351.80 | 2025-01-31 14:15:00 | 346.70 | STOP_HIT | 1.00 | 1.45% |
| SELL | retest2 | 2025-02-10 09:15:00 | 328.55 | 2025-02-11 11:15:00 | 312.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 328.55 | 2025-02-12 10:15:00 | 295.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-27 11:15:00 | 279.85 | 2025-03-03 10:15:00 | 265.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:15:00 | 279.85 | 2025-03-03 14:15:00 | 268.35 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-03-17 11:45:00 | 274.70 | 2025-03-19 10:15:00 | 278.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-03-18 10:45:00 | 275.10 | 2025-03-19 10:15:00 | 278.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-18 11:45:00 | 275.15 | 2025-03-19 10:15:00 | 278.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-03-18 15:15:00 | 275.00 | 2025-03-19 10:15:00 | 278.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-26 09:15:00 | 290.00 | 2025-03-26 14:15:00 | 287.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-03-26 09:45:00 | 290.70 | 2025-03-26 14:15:00 | 287.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-04-01 09:30:00 | 292.55 | 2025-04-02 09:15:00 | 290.25 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-04-01 10:00:00 | 292.70 | 2025-04-04 09:15:00 | 289.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-04-01 10:30:00 | 292.50 | 2025-04-04 11:15:00 | 288.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-04-01 12:00:00 | 292.70 | 2025-04-04 11:15:00 | 288.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-04-01 15:00:00 | 294.00 | 2025-04-04 11:15:00 | 288.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-02 15:00:00 | 294.00 | 2025-04-04 11:15:00 | 288.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-17 11:30:00 | 313.00 | 2025-04-25 10:15:00 | 313.75 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-04-17 14:30:00 | 312.85 | 2025-04-25 10:15:00 | 313.75 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-04-25 10:00:00 | 317.15 | 2025-04-25 10:15:00 | 313.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-20 09:30:00 | 357.15 | 2025-05-20 10:15:00 | 346.95 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-05-23 09:15:00 | 339.55 | 2025-06-03 09:15:00 | 332.80 | STOP_HIT | 1.00 | 1.99% |
| BUY | retest2 | 2025-06-11 14:30:00 | 336.25 | 2025-06-12 13:15:00 | 330.50 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-06-12 10:00:00 | 336.15 | 2025-06-12 13:15:00 | 330.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-06-17 11:15:00 | 329.25 | 2025-06-19 13:15:00 | 312.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 329.25 | 2025-06-20 09:15:00 | 318.30 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-06-17 15:15:00 | 327.90 | 2025-06-25 09:15:00 | 322.35 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2025-07-01 11:00:00 | 334.75 | 2025-07-07 09:15:00 | 334.05 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-07-01 12:00:00 | 334.30 | 2025-07-07 10:15:00 | 333.95 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-01 14:15:00 | 334.60 | 2025-07-07 10:15:00 | 333.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-07-02 09:30:00 | 337.35 | 2025-07-07 10:15:00 | 333.95 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-07 09:15:00 | 335.40 | 2025-07-07 10:15:00 | 333.95 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-09 14:30:00 | 328.25 | 2025-07-16 13:15:00 | 331.15 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-10 09:30:00 | 327.85 | 2025-07-16 13:15:00 | 331.15 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-11 09:45:00 | 328.80 | 2025-07-16 13:15:00 | 331.15 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-30 11:15:00 | 322.00 | 2025-08-04 09:15:00 | 305.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 319.50 | 2025-08-04 09:15:00 | 303.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:15:00 | 322.00 | 2025-08-04 15:15:00 | 307.60 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-07-31 09:15:00 | 319.50 | 2025-08-04 15:15:00 | 307.60 | STOP_HIT | 0.50 | 3.72% |
| BUY | retest2 | 2025-09-19 09:15:00 | 319.00 | 2025-09-19 14:15:00 | 314.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-19 11:30:00 | 317.20 | 2025-09-19 14:15:00 | 314.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-19 13:00:00 | 317.20 | 2025-09-19 14:15:00 | 314.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-19 13:30:00 | 317.15 | 2025-09-19 14:15:00 | 314.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-09 13:00:00 | 314.70 | 2025-10-10 10:15:00 | 320.40 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-09 13:30:00 | 314.70 | 2025-10-10 10:15:00 | 320.40 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-23 10:45:00 | 311.25 | 2025-10-23 15:15:00 | 314.15 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-23 11:45:00 | 311.00 | 2025-10-23 15:15:00 | 314.15 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-11-04 11:45:00 | 309.55 | 2025-11-14 09:15:00 | 301.85 | STOP_HIT | 1.00 | 2.49% |
| SELL | retest2 | 2025-11-04 12:30:00 | 309.10 | 2025-11-14 09:15:00 | 301.85 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-11-17 13:15:00 | 304.80 | 2025-11-18 13:15:00 | 299.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-11-27 14:00:00 | 294.10 | 2025-12-02 09:15:00 | 307.05 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-11-27 15:15:00 | 294.00 | 2025-12-02 09:15:00 | 307.05 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2025-12-03 09:15:00 | 299.20 | 2025-12-04 13:15:00 | 296.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-03 13:30:00 | 299.30 | 2025-12-04 13:15:00 | 296.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-03 14:00:00 | 300.00 | 2025-12-04 13:15:00 | 296.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-15 11:15:00 | 287.40 | 2025-12-16 10:15:00 | 283.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-15 14:15:00 | 287.65 | 2025-12-16 10:15:00 | 283.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-26 10:45:00 | 296.85 | 2025-12-29 11:15:00 | 293.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-26 11:15:00 | 296.85 | 2025-12-29 11:15:00 | 293.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-26 13:00:00 | 297.50 | 2025-12-29 11:15:00 | 293.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-02-03 10:15:00 | 300.40 | 2026-02-03 12:15:00 | 301.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-11 15:00:00 | 315.00 | 2026-02-12 09:15:00 | 309.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-02-16 10:30:00 | 308.70 | 2026-02-23 15:15:00 | 303.15 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2026-02-25 09:15:00 | 305.25 | 2026-03-02 09:15:00 | 298.70 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-02-26 09:15:00 | 306.50 | 2026-03-02 09:15:00 | 298.70 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2026-02-27 09:45:00 | 305.25 | 2026-03-02 09:15:00 | 298.70 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-03-18 09:15:00 | 262.30 | 2026-03-23 09:15:00 | 249.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:30:00 | 262.50 | 2026-03-23 09:15:00 | 249.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 09:15:00 | 262.30 | 2026-03-23 12:15:00 | 236.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:30:00 | 262.50 | 2026-03-23 12:15:00 | 236.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-09 11:30:00 | 241.31 | 2026-04-13 09:15:00 | 236.15 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2026-04-09 15:15:00 | 241.30 | 2026-04-13 09:15:00 | 236.15 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-10 12:30:00 | 241.38 | 2026-04-13 09:15:00 | 236.15 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-04-10 14:00:00 | 241.44 | 2026-04-13 09:15:00 | 236.15 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-04-29 09:15:00 | 284.75 | 2026-05-08 12:15:00 | 288.50 | STOP_HIT | 1.00 | 1.32% |
