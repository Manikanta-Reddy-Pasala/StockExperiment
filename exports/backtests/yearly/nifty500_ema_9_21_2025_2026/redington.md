# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 223.29
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 52 |
| ALERT1 | 42 |
| ALERT2 | 43 |
| ALERT2_SKIP | 23 |
| ALERT3 | 96 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 41 |
| PARTIAL | 14 |
| TARGET_HIT | 3 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 25
- **Target hits / Stop hits / Partials:** 3 / 39 / 14
- **Avg / median % per leg:** 1.76% / 3.14%
- **Sum % (uncompounded):** 98.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 5 | 26.3% | 3 | 16 | 0 | 0.60% | 11.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.14% | -0.1% |
| BUY @ 3rd Alert (retest2) | 18 | 5 | 27.8% | 3 | 15 | 0 | 0.65% | 11.6% |
| SELL (all) | 37 | 26 | 70.3% | 0 | 23 | 14 | 2.36% | 87.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 26 | 70.3% | 0 | 23 | 14 | 2.36% | 87.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.14% | -0.1% |
| retest2 (combined) | 55 | 31 | 56.4% | 3 | 38 | 14 | 1.80% | 98.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 276.99 | 283.57 | 283.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 275.85 | 282.03 | 283.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 267.56 | 264.42 | 267.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 269.22 | 265.38 | 268.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 269.22 | 265.38 | 268.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 269.84 | 266.27 | 268.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 269.52 | 266.27 | 268.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 267.40 | 266.50 | 268.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:00:00 | 266.10 | 266.78 | 267.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:45:00 | 266.46 | 266.41 | 267.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 265.98 | 266.41 | 267.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 269.50 | 265.06 | 264.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 269.50 | 265.06 | 264.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 270.10 | 266.67 | 265.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 268.60 | 270.36 | 268.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 268.65 | 270.02 | 268.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 273.25 | 270.02 | 268.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 270.05 | 269.90 | 268.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 270.05 | 270.11 | 268.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:00:00 | 270.95 | 270.11 | 268.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 268.50 | 269.79 | 268.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 267.85 | 269.79 | 268.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 267.70 | 269.37 | 268.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 267.70 | 269.37 | 268.83 | SL hit (close<static) qty=1.00 sl=267.80 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 293.00 | 294.63 | 294.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 291.55 | 294.01 | 294.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 294.05 | 294.02 | 294.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 294.95 | 294.21 | 294.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 294.95 | 294.21 | 294.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 294.45 | 294.25 | 294.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 301.80 | 294.25 | 294.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 302.25 | 295.85 | 295.11 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 293.85 | 296.66 | 296.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 292.95 | 295.91 | 296.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 301.95 | 292.53 | 293.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 300.15 | 294.06 | 293.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 305.70 | 299.27 | 296.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 301.40 | 304.00 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 297.70 | 302.74 | 300.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 297.70 | 302.74 | 300.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 297.40 | 301.67 | 300.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 298.45 | 301.67 | 300.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:00:00 | 298.55 | 300.47 | 300.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 14:15:00 | 297.15 | 299.80 | 299.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 14:15:00 | 297.15 | 299.80 | 299.83 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 302.45 | 300.11 | 299.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 12:15:00 | 304.25 | 300.85 | 300.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 315.20 | 325.70 | 321.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 316.20 | 323.80 | 321.31 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 314.25 | 319.50 | 319.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 312.20 | 318.04 | 319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 311.65 | 311.57 | 313.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 311.65 | 311.57 | 313.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 312.60 | 311.92 | 313.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 309.45 | 311.86 | 312.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 306.40 | 301.70 | 301.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 306.40 | 301.70 | 301.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 311.15 | 308.71 | 307.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 303.70 | 307.06 | 307.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 301.70 | 304.83 | 306.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 308.75 | 306.71 | 306.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 14:15:00 | 311.35 | 307.64 | 306.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 303.90 | 307.64 | 307.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 303.00 | 304.80 | 306.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 288.30 | 287.59 | 290.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 283.40 | 286.75 | 290.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 283.40 | 286.75 | 290.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 240.65 | 241.75 | 243.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 15:15:00 | 241.00 | 238.76 | 238.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 241.00 | 238.76 | 238.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 242.45 | 239.94 | 239.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 244.80 | 245.04 | 243.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:30:00 | 245.30 | 245.04 | 243.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 243.60 | 244.76 | 243.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 243.60 | 244.76 | 243.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 242.10 | 244.22 | 243.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 242.10 | 244.22 | 243.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 242.20 | 243.82 | 243.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:30:00 | 242.15 | 243.82 | 243.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 243.75 | 243.46 | 243.22 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 241.90 | 243.04 | 243.16 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 244.30 | 243.39 | 243.30 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 239.40 | 242.54 | 242.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 237.10 | 241.45 | 242.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 236.35 | 236.01 | 238.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:00:00 | 236.35 | 236.01 | 238.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 240.55 | 236.60 | 237.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 240.35 | 236.60 | 237.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 241.20 | 237.52 | 238.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 241.50 | 237.52 | 238.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 240.95 | 238.77 | 238.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 241.95 | 240.42 | 239.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 243.07 | 243.24 | 241.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 243.07 | 243.24 | 241.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 241.81 | 242.95 | 241.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 241.81 | 242.95 | 241.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 241.20 | 242.60 | 241.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 241.20 | 242.60 | 241.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 240.55 | 242.19 | 241.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 241.07 | 242.19 | 241.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 240.50 | 241.85 | 241.50 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 238.47 | 240.74 | 241.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 236.49 | 239.89 | 240.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 241.69 | 239.35 | 240.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 240.08 | 239.49 | 240.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:15:00 | 240.27 | 239.49 | 240.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 241.00 | 239.80 | 240.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 241.24 | 239.80 | 240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 242.84 | 240.40 | 240.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 242.84 | 240.40 | 240.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 242.05 | 240.73 | 240.59 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 238.80 | 240.61 | 240.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 237.80 | 240.05 | 240.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 244.73 | 239.52 | 240.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 243.01 | 240.22 | 240.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 243.46 | 240.22 | 240.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 242.46 | 240.67 | 240.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 243.49 | 241.92 | 241.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 243.93 | 244.21 | 243.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:30:00 | 243.66 | 244.21 | 243.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 244.00 | 244.17 | 243.14 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 242.00 | 242.82 | 242.87 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 244.58 | 243.08 | 242.98 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 241.25 | 242.99 | 243.15 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 266.00 | 246.92 | 244.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 275.70 | 252.68 | 247.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 287.20 | 287.28 | 279.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:30:00 | 287.04 | 287.28 | 279.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 293.95 | 296.86 | 293.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 293.81 | 296.86 | 293.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 292.06 | 295.90 | 293.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 292.41 | 295.90 | 293.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 290.05 | 294.73 | 292.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 290.05 | 294.73 | 292.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 285.07 | 290.57 | 291.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 280.34 | 285.82 | 288.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 264.67 | 266.18 | 269.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 293.01 | 273.40 | 272.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 293.01 | 273.40 | 272.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 15:15:00 | 295.00 | 277.72 | 274.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 276.15 | 278.43 | 278.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 275.15 | 276.53 | 277.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 277.15 | 276.51 | 277.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 277.85 | 276.78 | 277.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 277.20 | 276.78 | 277.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 276.75 | 276.77 | 277.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 277.45 | 276.77 | 277.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 277.30 | 276.88 | 277.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 277.35 | 276.88 | 277.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 276.80 | 276.86 | 277.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 277.95 | 276.86 | 277.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 274.00 | 276.29 | 276.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 273.95 | 276.29 | 276.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 285.00 | 272.82 | 273.59 | SL hit (close>static) qty=1.00 sl=277.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 286.55 | 275.57 | 274.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 291.25 | 280.68 | 277.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 274.00 | 279.42 | 279.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 271.65 | 277.87 | 278.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 272.70 | 275.22 | 275.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 272.60 | 274.44 | 275.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 272.20 | 273.66 | 274.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:15:00 | 259.06 | 261.43 | 262.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:15:00 | 258.97 | 260.88 | 261.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:15:00 | 258.59 | 260.88 | 261.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 256.30 | 255.86 | 257.90 | SL hit (close>ema200) qty=0.50 sl=255.86 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 277.45 | 256.63 | 255.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 282.30 | 261.77 | 258.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 11:15:00 | 289.70 | 290.40 | 282.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 289.70 | 290.40 | 282.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 291.15 | 294.93 | 294.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 291.15 | 294.93 | 294.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 293.80 | 294.70 | 294.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 294.30 | 294.77 | 294.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 295.05 | 294.77 | 294.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 289.90 | 294.80 | 294.76 | SL hit (close<static) qty=1.00 sl=290.50 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 290.50 | 293.94 | 294.38 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 297.60 | 294.28 | 293.83 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 286.80 | 293.51 | 293.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 285.30 | 289.62 | 291.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 297.10 | 288.97 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 290.20 | 289.21 | 290.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 290.20 | 289.21 | 290.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 293.50 | 290.07 | 291.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 295.60 | 290.07 | 291.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 290.60 | 290.18 | 291.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 289.75 | 290.18 | 291.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 289.80 | 290.44 | 291.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 289.20 | 290.63 | 291.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 288.65 | 289.50 | 290.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 288.65 | 289.11 | 289.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 285.75 | 287.84 | 288.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 275.26 | 278.51 | 280.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 275.31 | 278.51 | 280.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 15:15:00 | 274.74 | 277.34 | 279.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 15:15:00 | 274.22 | 277.34 | 279.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 279.60 | 277.80 | 279.49 | SL hit (close>ema200) qty=0.50 sl=277.80 alert=retest2 |

### Cycle 36 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 276.40 | 270.84 | 270.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 278.00 | 275.19 | 273.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 281.00 | 281.09 | 279.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 276.60 | 281.09 | 279.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 275.75 | 280.02 | 279.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 275.75 | 280.02 | 279.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 275.15 | 279.04 | 278.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 275.15 | 279.04 | 278.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 275.00 | 278.24 | 278.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 272.60 | 277.11 | 277.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 275.35 | 274.71 | 276.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 275.35 | 274.71 | 276.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 271.95 | 269.86 | 271.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 271.95 | 269.86 | 271.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 272.35 | 270.36 | 271.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 272.45 | 270.36 | 271.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 272.70 | 270.83 | 271.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 274.45 | 270.83 | 271.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 276.85 | 272.03 | 272.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 276.85 | 272.03 | 272.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 275.55 | 272.74 | 272.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 277.70 | 275.19 | 274.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 274.95 | 275.14 | 274.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 274.95 | 275.14 | 274.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 273.85 | 274.88 | 274.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 273.85 | 274.88 | 274.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 274.20 | 274.75 | 274.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 273.75 | 274.75 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 274.10 | 274.62 | 274.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 273.95 | 274.62 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 274.65 | 274.62 | 274.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 274.05 | 274.62 | 274.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 273.65 | 274.43 | 274.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 273.65 | 274.43 | 274.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 273.20 | 274.18 | 274.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:15:00 | 273.00 | 274.18 | 274.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 272.35 | 273.82 | 274.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 271.65 | 273.09 | 273.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 274.40 | 267.37 | 269.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 267.75 | 267.45 | 268.97 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 273.30 | 270.15 | 269.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 274.70 | 271.97 | 270.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 277.20 | 278.15 | 276.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:15:00 | 285.60 | 278.15 | 276.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 284.15 | 286.06 | 284.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 284.15 | 286.06 | 284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 285.10 | 285.86 | 284.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 294.35 | 285.86 | 284.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 285.20 | 287.12 | 285.34 | SL hit (close<ema400) qty=1.00 sl=285.34 alert=retest1 |

### Cycle 41 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 279.95 | 283.69 | 284.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 275.25 | 282.00 | 283.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 272.60 | 271.26 | 274.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 272.60 | 271.26 | 274.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 273.70 | 271.75 | 274.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 273.95 | 271.75 | 274.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 273.95 | 272.59 | 273.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 268.70 | 273.77 | 274.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 255.26 | 259.96 | 261.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 252.90 | 251.70 | 254.46 | SL hit (close>ema200) qty=0.50 sl=251.70 alert=retest2 |

### Cycle 42 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 261.00 | 256.17 | 255.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 261.85 | 257.68 | 256.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 266.00 | 267.87 | 264.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 266.00 | 267.08 | 265.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 266.00 | 267.08 | 265.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 262.50 | 266.16 | 265.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 262.50 | 266.16 | 265.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 263.00 | 265.53 | 264.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:00:00 | 267.15 | 265.41 | 264.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 268.00 | 272.42 | 272.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 268.00 | 272.42 | 272.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 264.60 | 268.78 | 270.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 268.60 | 268.06 | 269.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 268.60 | 268.06 | 269.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 270.25 | 268.70 | 269.67 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 272.15 | 270.49 | 270.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 274.40 | 271.33 | 270.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 272.65 | 273.24 | 271.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 272.65 | 273.24 | 271.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 270.35 | 272.66 | 271.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 270.35 | 272.66 | 271.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 270.70 | 272.27 | 271.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 273.00 | 272.27 | 271.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 271.70 | 271.68 | 271.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 271.50 | 271.71 | 271.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 270.15 | 271.40 | 271.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 270.15 | 271.40 | 271.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 267.35 | 269.70 | 270.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 259.45 | 258.40 | 261.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:30:00 | 260.00 | 258.40 | 261.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 259.75 | 258.83 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 259.75 | 258.83 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 260.90 | 259.43 | 260.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 260.90 | 259.43 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 261.00 | 259.75 | 260.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 259.40 | 259.75 | 260.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 12:15:00 | 246.43 | 249.88 | 252.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 244.10 | 242.46 | 245.79 | SL hit (close>ema200) qty=0.50 sl=242.46 alert=retest2 |

### Cycle 46 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 270.15 | 248.41 | 245.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 274.00 | 253.53 | 248.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 248.65 | 258.36 | 259.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 11:15:00 | 246.80 | 250.68 | 254.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 250.00 | 249.13 | 252.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 251.60 | 249.13 | 252.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 249.20 | 249.14 | 252.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 247.35 | 248.84 | 251.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 247.60 | 248.39 | 250.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 247.30 | 247.98 | 250.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 234.98 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 235.22 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 234.94 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 261.85 | 240.58 | 243.03 | SL hit (close>ema200) qty=0.50 sl=240.58 alert=retest2 |

### Cycle 48 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 262.60 | 244.98 | 244.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 266.25 | 249.24 | 246.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 242.50 | 248.41 | 249.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 235.30 | 239.87 | 243.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 233.05 | 232.43 | 236.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 233.50 | 232.43 | 236.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 229.90 | 229.09 | 231.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 224.60 | 230.25 | 231.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 213.37 | 216.45 | 220.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 209.85 | 208.58 | 212.53 | SL hit (close>ema200) qty=0.50 sl=208.58 alert=retest2 |

### Cycle 50 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 205.79 | 201.20 | 201.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 208.25 | 202.61 | 201.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 218.39 | 218.61 | 215.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 218.39 | 218.61 | 215.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 218.40 | 220.52 | 218.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 218.83 | 220.32 | 218.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 218.82 | 219.61 | 218.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 223.51 | 219.16 | 218.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 240.71 | 227.88 | 224.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 228.00 | 228.50 | 228.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 226.20 | 228.04 | 228.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 217.50 | 219.07 | 219.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 216.58 | 218.48 | 219.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 221.17 | 216.63 | 216.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 221.17 | 216.63 | 216.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 223.34 | 220.64 | 219.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 222.36 | 223.30 | 221.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 222.36 | 223.30 | 221.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 221.80 | 223.00 | 221.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 222.00 | 223.00 | 221.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 222.96 | 222.99 | 221.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 222.26 | 222.99 | 221.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:15:00 | 292.42 | 2025-05-22 11:15:00 | 276.99 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-05-29 11:00:00 | 266.10 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-05-29 12:45:00 | 266.46 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-05-29 13:15:00 | 265.98 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-06-04 09:15:00 | 273.25 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-04 11:45:00 | 270.05 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-04 12:30:00 | 270.05 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-04 13:00:00 | 270.95 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-05 09:15:00 | 268.75 | 2025-06-11 09:15:00 | 295.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 12:15:00 | 298.45 | 2025-06-26 14:15:00 | 297.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-26 14:00:00 | 298.55 | 2025-06-26 14:15:00 | 297.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-07 15:15:00 | 309.45 | 2025-07-14 10:15:00 | 306.40 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-08-11 09:15:00 | 240.65 | 2025-08-18 15:15:00 | 241.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-09-29 11:30:00 | 264.67 | 2025-09-29 14:15:00 | 293.01 | STOP_HIT | 1.00 | -10.71% |
| SELL | retest2 | 2025-10-08 15:15:00 | 273.95 | 2025-10-10 09:15:00 | 285.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-10-16 10:15:00 | 272.70 | 2025-10-30 11:15:00 | 259.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 12:15:00 | 272.60 | 2025-10-30 12:15:00 | 258.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 13:30:00 | 272.20 | 2025-10-30 12:15:00 | 258.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:15:00 | 272.70 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 6.01% |
| SELL | retest2 | 2025-10-16 12:15:00 | 272.60 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2025-10-16 13:30:00 | 272.20 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 5.84% |
| BUY | retest2 | 2025-11-14 14:30:00 | 294.30 | 2025-11-18 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-11-14 15:00:00 | 295.05 | 2025-11-18 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-24 13:15:00 | 289.75 | 2025-12-03 12:15:00 | 275.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 14:15:00 | 289.80 | 2025-12-03 12:15:00 | 275.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 15:15:00 | 289.20 | 2025-12-03 15:15:00 | 274.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:30:00 | 288.65 | 2025-12-03 15:15:00 | 274.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 13:15:00 | 289.75 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-11-24 14:15:00 | 289.80 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2025-11-24 15:15:00 | 289.20 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-25 14:30:00 | 288.65 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-11-27 13:15:00 | 285.75 | 2025-12-05 09:15:00 | 271.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 285.75 | 2025-12-08 14:15:00 | 267.30 | STOP_HIT | 0.50 | 6.46% |
| BUY | retest1 | 2026-01-06 09:15:00 | 285.60 | 2026-01-08 11:15:00 | 285.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-01-08 09:15:00 | 294.35 | 2026-01-08 13:15:00 | 282.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-01-16 09:15:00 | 268.70 | 2026-01-23 10:15:00 | 255.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 268.70 | 2026-01-27 15:15:00 | 252.90 | STOP_HIT | 0.50 | 5.88% |
| BUY | retest2 | 2026-02-02 14:00:00 | 267.15 | 2026-02-05 12:15:00 | 268.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-11 09:15:00 | 273.00 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-11 10:45:00 | 271.70 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-11 11:45:00 | 271.50 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-18 09:15:00 | 259.40 | 2026-02-23 12:15:00 | 246.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:15:00 | 259.40 | 2026-02-25 09:15:00 | 244.10 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2026-03-06 10:45:00 | 247.35 | 2026-03-09 09:15:00 | 234.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 247.60 | 2026-03-09 09:15:00 | 235.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 247.30 | 2026-03-09 09:15:00 | 234.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 247.35 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.86% |
| SELL | retest2 | 2026-03-06 13:15:00 | 247.60 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.76% |
| SELL | retest2 | 2026-03-06 14:30:00 | 247.30 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.88% |
| SELL | retest2 | 2026-03-19 09:15:00 | 224.60 | 2026-03-23 09:15:00 | 213.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 224.60 | 2026-03-24 12:15:00 | 209.85 | STOP_HIT | 0.50 | 6.57% |
| BUY | retest2 | 2026-04-13 10:30:00 | 218.83 | 2026-04-17 09:15:00 | 240.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 218.82 | 2026-04-17 09:15:00 | 240.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 223.51 | 2026-04-21 15:15:00 | 228.00 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2026-04-28 11:15:00 | 217.50 | 2026-05-04 10:15:00 | 221.17 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-04-28 11:45:00 | 216.58 | 2026-05-04 10:15:00 | 221.17 | STOP_HIT | 1.00 | -2.12% |
