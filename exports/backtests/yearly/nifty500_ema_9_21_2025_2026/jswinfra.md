# JSW Infrastructure Ltd. (JSWINFRA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 284.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 46 |
| ALERT2 | 46 |
| ALERT2_SKIP | 27 |
| ALERT3 | 112 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 48 |
| PARTIAL | 10 |
| TARGET_HIT | 0 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 36
- **Target hits / Stop hits / Partials:** 0 / 51 / 10
- **Avg / median % per leg:** 0.71% / -0.93%
- **Sum % (uncompounded):** 43.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 0 | 10 | 0 | -0.89% | -8.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 0 | 10 | 0 | -0.89% | -8.9% |
| SELL (all) | 51 | 24 | 47.1% | 0 | 41 | 10 | 1.02% | 52.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.26% | -3.8% |
| SELL @ 3rd Alert (retest2) | 48 | 24 | 50.0% | 0 | 38 | 10 | 1.17% | 56.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.26% | -3.8% |
| retest2 (combined) | 58 | 25 | 43.1% | 0 | 48 | 10 | 0.81% | 47.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 293.60 | 288.96 | 288.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 296.15 | 292.42 | 290.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 293.80 | 294.13 | 292.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 293.80 | 294.13 | 292.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 296.10 | 297.42 | 296.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 292.40 | 297.42 | 296.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 291.00 | 296.14 | 295.70 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 10:15:00 | 288.75 | 294.66 | 295.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 287.30 | 289.19 | 290.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 284.85 | 284.75 | 287.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 285.35 | 284.75 | 287.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 286.10 | 285.18 | 286.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 286.10 | 285.18 | 286.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 288.20 | 285.57 | 286.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:45:00 | 287.20 | 285.57 | 286.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 288.00 | 286.06 | 286.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 287.00 | 286.06 | 286.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 287.65 | 286.74 | 286.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 287.65 | 286.74 | 286.71 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 286.30 | 286.67 | 286.69 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 288.95 | 287.13 | 286.90 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 285.20 | 286.71 | 286.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 285.10 | 286.11 | 286.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 14:15:00 | 286.55 | 286.01 | 286.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 14:15:00 | 286.55 | 286.01 | 286.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 286.55 | 286.01 | 286.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 286.55 | 286.01 | 286.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 286.85 | 286.18 | 286.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 288.75 | 286.18 | 286.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 289.95 | 286.93 | 286.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 10:15:00 | 292.55 | 288.64 | 287.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 15:15:00 | 289.70 | 289.76 | 288.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:15:00 | 291.00 | 289.76 | 288.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 290.50 | 290.77 | 289.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 290.30 | 290.77 | 289.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 291.50 | 290.91 | 290.03 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 286.90 | 289.50 | 289.84 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 292.25 | 289.96 | 289.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 294.50 | 290.87 | 290.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 304.20 | 304.28 | 299.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:00:00 | 304.20 | 304.28 | 299.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 309.00 | 311.64 | 310.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 310.00 | 311.64 | 310.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 311.10 | 311.53 | 310.63 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 307.45 | 310.10 | 310.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 303.95 | 308.62 | 309.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 304.00 | 303.85 | 305.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 300.25 | 303.85 | 305.88 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 11:45:00 | 301.35 | 302.39 | 304.59 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 304.50 | 302.91 | 304.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 304.50 | 302.91 | 304.45 | SL hit (close>ema400) qty=1.00 sl=304.45 alert=retest1 |

### Cycle 11 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 310.75 | 305.42 | 305.04 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 302.80 | 305.69 | 305.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 298.45 | 304.24 | 305.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 301.50 | 299.44 | 301.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 301.50 | 299.44 | 301.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 301.50 | 299.44 | 301.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 301.25 | 299.44 | 301.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 301.60 | 299.88 | 301.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 302.65 | 299.88 | 301.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 304.65 | 300.83 | 301.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 304.65 | 300.83 | 301.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 303.40 | 301.34 | 301.83 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 304.30 | 302.45 | 302.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 309.50 | 304.39 | 303.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 308.40 | 308.45 | 306.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 308.40 | 308.45 | 306.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 314.00 | 315.61 | 314.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 314.00 | 315.61 | 314.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 312.20 | 314.93 | 313.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 312.50 | 314.93 | 313.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 311.40 | 314.22 | 313.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 311.40 | 314.22 | 313.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 310.90 | 313.03 | 313.24 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 313.95 | 313.25 | 313.19 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 309.85 | 312.57 | 312.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 309.00 | 311.85 | 312.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 311.20 | 310.56 | 311.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 15:15:00 | 311.20 | 310.56 | 311.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 311.20 | 310.56 | 311.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 306.85 | 310.56 | 311.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 309.70 | 306.69 | 306.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 309.70 | 306.69 | 306.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 310.60 | 307.47 | 307.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 311.50 | 311.89 | 310.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 10:15:00 | 311.50 | 311.89 | 310.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 311.50 | 311.89 | 310.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 315.25 | 311.93 | 311.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 13:15:00 | 311.60 | 314.47 | 314.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 311.60 | 314.47 | 314.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 14:15:00 | 310.30 | 313.64 | 314.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 11:15:00 | 313.40 | 312.58 | 313.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 313.40 | 312.58 | 313.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 313.40 | 312.58 | 313.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 313.35 | 312.58 | 313.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 311.85 | 312.43 | 313.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 311.45 | 312.43 | 313.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 311.10 | 312.04 | 312.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:45:00 | 311.55 | 312.05 | 312.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 311.45 | 311.42 | 312.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 314.60 | 310.01 | 310.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 314.60 | 310.01 | 310.73 | SL hit (close>static) qty=1.00 sl=313.60 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 314.70 | 311.74 | 311.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 316.90 | 313.54 | 312.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 319.20 | 321.10 | 319.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 10:15:00 | 319.20 | 321.10 | 319.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 319.20 | 321.10 | 319.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 319.20 | 321.10 | 319.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 318.90 | 320.66 | 319.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 318.70 | 320.66 | 319.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 319.65 | 320.46 | 319.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 321.10 | 320.47 | 319.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 317.05 | 319.93 | 319.42 | SL hit (close<static) qty=1.00 sl=318.70 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 314.75 | 318.90 | 318.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 314.15 | 315.95 | 317.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 309.00 | 307.61 | 310.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:45:00 | 308.55 | 307.61 | 310.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 311.00 | 308.57 | 310.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 311.00 | 308.57 | 310.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 311.95 | 309.25 | 310.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 311.25 | 309.25 | 310.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 310.95 | 309.80 | 310.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 311.00 | 309.80 | 310.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 309.90 | 309.82 | 310.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:45:00 | 309.50 | 309.71 | 310.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 299.70 | 298.24 | 298.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 299.70 | 298.24 | 298.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 302.50 | 300.17 | 299.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 309.85 | 311.31 | 308.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:45:00 | 310.00 | 311.31 | 308.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 308.75 | 310.32 | 309.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:30:00 | 308.60 | 310.32 | 309.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 308.15 | 309.88 | 308.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 308.15 | 309.88 | 308.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 308.30 | 309.57 | 308.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 310.20 | 309.57 | 308.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 309.50 | 309.78 | 309.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 307.15 | 309.02 | 309.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 307.15 | 309.02 | 309.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 305.95 | 307.78 | 308.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 299.00 | 298.76 | 301.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:00:00 | 299.00 | 298.76 | 301.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 299.65 | 298.84 | 300.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 300.05 | 298.84 | 300.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 302.00 | 299.47 | 300.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 298.30 | 299.47 | 300.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 303.80 | 299.12 | 298.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 303.80 | 299.12 | 298.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 306.20 | 302.90 | 301.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 305.30 | 305.45 | 303.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 305.30 | 305.45 | 303.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 305.30 | 305.45 | 303.66 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 301.15 | 303.52 | 303.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 299.20 | 302.66 | 303.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 302.75 | 301.63 | 302.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 302.75 | 301.63 | 302.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 302.75 | 301.63 | 302.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 302.55 | 301.63 | 302.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 303.45 | 301.99 | 302.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:00:00 | 302.45 | 302.44 | 302.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 302.10 | 302.29 | 302.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 309.50 | 303.56 | 303.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 309.50 | 303.56 | 303.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 312.85 | 307.35 | 305.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 313.15 | 313.36 | 310.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 313.15 | 313.36 | 310.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 313.55 | 314.75 | 312.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 312.60 | 314.75 | 312.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 315.30 | 314.88 | 313.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 313.75 | 314.88 | 313.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 315.50 | 315.02 | 314.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 314.70 | 315.02 | 314.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 314.80 | 315.36 | 314.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:45:00 | 314.80 | 315.36 | 314.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 314.90 | 315.27 | 314.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 317.90 | 315.27 | 314.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 336.25 | 339.07 | 339.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 336.25 | 339.07 | 339.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 330.80 | 337.28 | 338.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 324.25 | 322.99 | 327.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 324.25 | 322.99 | 327.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 317.50 | 315.89 | 317.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 317.50 | 315.89 | 317.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 305.80 | 306.69 | 308.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 305.25 | 307.26 | 307.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 305.10 | 305.94 | 306.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 304.60 | 305.68 | 306.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 309.65 | 306.78 | 306.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 309.65 | 306.78 | 306.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 310.30 | 307.48 | 306.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 307.85 | 308.54 | 307.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 307.85 | 308.54 | 307.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 307.85 | 308.54 | 307.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 308.20 | 308.54 | 307.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 309.00 | 308.64 | 307.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 310.85 | 309.14 | 308.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 309.90 | 309.69 | 308.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 298.20 | 307.22 | 307.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 298.20 | 307.22 | 307.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 11:15:00 | 295.50 | 303.43 | 305.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 298.20 | 296.07 | 298.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 298.20 | 296.07 | 298.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 298.20 | 296.07 | 298.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 298.20 | 296.07 | 298.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 299.00 | 296.66 | 298.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 299.00 | 296.66 | 298.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 299.20 | 297.17 | 298.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 299.20 | 297.17 | 298.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 299.00 | 297.53 | 298.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:45:00 | 297.25 | 297.56 | 298.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 300.50 | 298.41 | 299.02 | SL hit (close>static) qty=1.00 sl=299.45 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 284.40 | 282.59 | 282.57 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 280.00 | 282.63 | 282.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 279.45 | 281.19 | 281.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 281.50 | 281.22 | 281.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 15:15:00 | 281.50 | 281.22 | 281.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 281.50 | 281.22 | 281.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 282.35 | 281.22 | 281.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 281.10 | 281.20 | 281.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 280.75 | 281.20 | 281.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:45:00 | 280.60 | 280.43 | 280.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 266.71 | 271.09 | 273.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 266.57 | 271.09 | 273.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 269.70 | 268.99 | 271.58 | SL hit (close>ema200) qty=0.50 sl=268.99 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 273.55 | 270.19 | 270.03 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 270.50 | 271.61 | 271.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 269.70 | 271.04 | 271.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 269.50 | 268.94 | 269.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 266.50 | 268.60 | 269.58 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 268.35 | 267.90 | 268.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 268.35 | 267.90 | 268.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 270.00 | 268.32 | 268.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 270.00 | 268.32 | 268.89 | SL hit (close>ema400) qty=1.00 sl=268.89 alert=retest1 |

### Cycle 33 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 266.65 | 265.78 | 265.72 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 264.70 | 265.56 | 265.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 264.45 | 265.34 | 265.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 266.30 | 265.48 | 265.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 266.30 | 265.48 | 265.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 266.30 | 265.48 | 265.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 266.30 | 265.48 | 265.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 265.40 | 265.46 | 265.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 266.10 | 265.46 | 265.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 267.10 | 265.79 | 265.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 271.50 | 267.18 | 266.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 272.25 | 272.74 | 271.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 272.25 | 272.74 | 271.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 270.80 | 272.32 | 271.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 269.95 | 272.32 | 271.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 270.60 | 271.98 | 271.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 269.70 | 271.98 | 271.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 269.70 | 271.21 | 271.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 268.95 | 270.53 | 271.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 272.10 | 270.55 | 270.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 272.10 | 270.55 | 270.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 272.10 | 270.55 | 270.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:30:00 | 272.30 | 270.55 | 270.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 273.30 | 271.10 | 271.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 14:15:00 | 274.30 | 272.23 | 271.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 285.70 | 286.54 | 283.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 285.70 | 286.54 | 283.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 284.25 | 285.01 | 284.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 281.45 | 285.01 | 284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 280.75 | 284.16 | 283.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 280.75 | 284.16 | 283.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 279.30 | 283.19 | 283.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 284.15 | 283.28 | 283.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 287.10 | 284.44 | 283.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 10:15:00 | 284.30 | 284.87 | 284.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 284.30 | 284.87 | 284.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 284.30 | 284.87 | 284.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 284.30 | 284.87 | 284.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 284.65 | 284.83 | 284.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 285.45 | 284.84 | 284.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 283.05 | 284.49 | 284.20 | SL hit (close<static) qty=1.00 sl=284.00 alert=retest2 |

### Cycle 40 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 283.25 | 284.16 | 284.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 283.10 | 283.84 | 284.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 284.95 | 284.06 | 284.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 284.95 | 284.06 | 284.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 284.95 | 284.06 | 284.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 284.90 | 284.06 | 284.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 283.25 | 283.90 | 284.03 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 285.15 | 284.18 | 284.14 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 283.30 | 284.22 | 284.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 281.60 | 283.42 | 283.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 277.80 | 276.82 | 278.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 277.80 | 276.82 | 278.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 277.80 | 276.82 | 278.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 278.80 | 276.82 | 278.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 275.90 | 276.64 | 278.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 278.00 | 276.64 | 278.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 268.00 | 265.77 | 268.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 265.30 | 265.51 | 267.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 273.40 | 262.06 | 262.75 | SL hit (close>static) qty=1.00 sl=271.70 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 276.20 | 264.89 | 263.97 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 263.55 | 268.11 | 268.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 260.60 | 264.50 | 266.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 263.45 | 261.06 | 263.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 263.45 | 261.06 | 263.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 263.45 | 261.06 | 263.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 262.85 | 261.06 | 263.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 260.40 | 260.93 | 263.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 262.05 | 260.93 | 263.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 261.30 | 260.17 | 261.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 261.30 | 260.17 | 261.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 262.70 | 260.67 | 261.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 262.70 | 260.67 | 261.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 263.35 | 261.21 | 261.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 263.35 | 261.21 | 261.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 261.40 | 261.25 | 261.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:30:00 | 261.25 | 261.24 | 261.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 263.95 | 260.68 | 260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 263.95 | 260.68 | 260.36 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 258.30 | 260.28 | 260.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 257.05 | 259.31 | 259.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 258.00 | 256.84 | 258.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 258.00 | 256.84 | 258.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 259.15 | 257.30 | 258.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 259.15 | 257.30 | 258.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 260.00 | 257.84 | 258.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 265.85 | 257.84 | 258.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 263.50 | 258.97 | 258.94 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 259.05 | 262.11 | 262.43 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 268.65 | 262.76 | 262.16 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 261.50 | 264.02 | 264.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 258.15 | 261.98 | 262.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 262.20 | 259.91 | 260.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 262.20 | 259.91 | 260.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 262.20 | 259.91 | 260.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:30:00 | 261.80 | 259.91 | 260.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 262.15 | 260.35 | 260.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 262.00 | 260.35 | 260.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 262.35 | 261.43 | 261.31 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 11:15:00 | 260.75 | 261.19 | 261.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 13:15:00 | 259.55 | 260.71 | 260.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 259.60 | 259.28 | 259.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 15:00:00 | 259.60 | 259.28 | 259.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 257.45 | 258.86 | 259.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 256.85 | 258.46 | 259.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 257.05 | 255.64 | 255.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 257.00 | 255.96 | 256.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 260.00 | 256.77 | 256.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 260.00 | 256.77 | 256.40 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 254.50 | 256.33 | 256.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 253.55 | 255.11 | 255.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 255.25 | 254.64 | 255.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 255.25 | 254.64 | 255.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 255.25 | 254.64 | 255.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 258.80 | 254.64 | 255.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 259.25 | 255.56 | 255.56 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 253.35 | 255.72 | 255.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 252.80 | 255.04 | 255.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 254.55 | 248.83 | 250.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 254.55 | 248.83 | 250.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 254.55 | 248.83 | 250.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 254.55 | 248.83 | 250.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 257.70 | 250.61 | 251.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 257.70 | 250.61 | 251.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 256.45 | 251.78 | 251.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 260.95 | 254.99 | 253.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 258.05 | 264.94 | 261.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 258.05 | 264.94 | 261.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 258.05 | 264.94 | 261.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 262.95 | 261.64 | 260.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 259.55 | 261.69 | 261.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 259.55 | 261.69 | 261.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 258.65 | 261.08 | 261.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 260.50 | 260.01 | 260.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 260.50 | 260.01 | 260.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 260.50 | 260.01 | 260.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 260.50 | 260.01 | 260.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 261.50 | 260.31 | 260.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 261.50 | 260.31 | 260.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 260.75 | 260.40 | 260.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 259.65 | 260.49 | 260.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:45:00 | 258.20 | 259.46 | 260.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 246.67 | 253.11 | 256.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 15:15:00 | 251.25 | 251.13 | 253.83 | SL hit (close>ema200) qty=0.50 sl=251.13 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 259.65 | 255.19 | 254.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 260.85 | 257.06 | 255.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 256.65 | 258.01 | 256.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 256.65 | 258.01 | 256.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 256.65 | 258.01 | 256.68 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 253.30 | 255.61 | 255.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 251.70 | 254.41 | 255.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 249.90 | 244.80 | 247.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 249.90 | 244.80 | 247.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 249.90 | 244.80 | 247.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 249.90 | 244.80 | 247.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 252.30 | 246.30 | 248.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 252.30 | 246.30 | 248.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 252.00 | 249.17 | 249.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 258.90 | 251.11 | 249.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 256.55 | 256.56 | 253.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 254.35 | 256.11 | 253.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 254.35 | 256.11 | 253.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 254.35 | 256.11 | 253.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 253.40 | 255.57 | 253.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 253.40 | 255.57 | 253.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 253.65 | 255.19 | 253.91 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 249.90 | 253.07 | 253.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 244.85 | 251.43 | 252.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 247.88 | 245.32 | 248.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 247.88 | 245.32 | 248.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 247.88 | 245.32 | 248.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 240.33 | 246.96 | 247.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 251.34 | 240.26 | 239.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 251.34 | 240.26 | 239.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 252.74 | 245.76 | 242.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 261.43 | 262.83 | 259.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 261.43 | 262.83 | 259.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 272.14 | 273.31 | 271.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:15:00 | 276.12 | 273.19 | 271.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 274.71 | 276.43 | 276.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 274.71 | 276.43 | 276.44 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 279.20 | 276.68 | 276.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 11:15:00 | 279.97 | 277.72 | 277.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 284.60 | 285.13 | 283.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 284.60 | 285.13 | 283.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 281.77 | 284.44 | 283.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 281.15 | 284.44 | 283.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 279.84 | 283.52 | 282.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 280.00 | 283.52 | 282.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 277.00 | 282.21 | 282.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 273.98 | 279.67 | 281.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 273.98 | 272.26 | 275.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 14:00:00 | 273.98 | 272.26 | 275.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 270.79 | 271.96 | 275.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 274.90 | 271.96 | 275.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 273.00 | 272.30 | 274.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:45:00 | 272.35 | 273.39 | 274.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:00:00 | 272.15 | 273.06 | 273.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:00:00 | 272.40 | 272.93 | 273.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 271.55 | 272.93 | 273.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 272.80 | 272.51 | 273.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 274.10 | 272.51 | 273.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 274.00 | 272.81 | 273.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 277.60 | 272.81 | 273.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 277.50 | 273.75 | 273.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 277.50 | 273.75 | 273.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 281.85 | 276.46 | 275.02 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 09:15:00 | 287.00 | 2025-05-23 14:15:00 | 287.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-16 09:15:00 | 300.25 | 2025-06-16 13:15:00 | 304.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest1 | 2025-06-16 11:45:00 | 301.35 | 2025-06-16 13:15:00 | 304.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-03 09:15:00 | 306.85 | 2025-07-08 11:15:00 | 309.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-11 09:15:00 | 315.25 | 2025-07-15 13:15:00 | 311.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-16 13:15:00 | 311.45 | 2025-07-21 09:15:00 | 314.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-17 09:45:00 | 311.10 | 2025-07-21 09:15:00 | 314.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-17 14:45:00 | 311.55 | 2025-07-21 09:15:00 | 314.60 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-18 09:30:00 | 311.45 | 2025-07-21 09:15:00 | 314.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-24 15:00:00 | 321.10 | 2025-07-25 09:15:00 | 317.05 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-30 12:45:00 | 309.50 | 2025-08-13 12:15:00 | 299.70 | STOP_HIT | 1.00 | 3.17% |
| BUY | retest2 | 2025-08-21 09:15:00 | 310.20 | 2025-08-22 11:15:00 | 307.15 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-22 10:00:00 | 309.50 | 2025-08-22 11:15:00 | 307.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-29 09:15:00 | 298.30 | 2025-09-02 09:15:00 | 303.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-08 14:00:00 | 302.45 | 2025-09-09 09:15:00 | 309.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-09-08 14:45:00 | 302.10 | 2025-09-09 09:15:00 | 309.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-09-16 09:15:00 | 317.90 | 2025-09-25 13:15:00 | 336.25 | STOP_HIT | 1.00 | 5.77% |
| SELL | retest2 | 2025-10-13 09:15:00 | 305.25 | 2025-10-14 15:15:00 | 309.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-14 11:15:00 | 305.10 | 2025-10-14 15:15:00 | 309.65 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-14 12:00:00 | 304.60 | 2025-10-14 15:15:00 | 309.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-16 09:45:00 | 310.85 | 2025-10-17 09:15:00 | 298.20 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-10-16 14:30:00 | 309.90 | 2025-10-17 09:15:00 | 298.20 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-10-23 11:45:00 | 297.25 | 2025-10-23 13:15:00 | 300.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-24 10:00:00 | 296.80 | 2025-11-06 13:15:00 | 281.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 10:00:00 | 296.95 | 2025-11-06 13:15:00 | 282.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 09:30:00 | 297.20 | 2025-11-06 13:15:00 | 282.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 12:00:00 | 294.80 | 2025-11-07 09:15:00 | 280.25 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-10-30 13:15:00 | 295.00 | 2025-11-07 09:15:00 | 280.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 10:00:00 | 296.80 | 2025-11-07 11:15:00 | 284.75 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-10-27 10:00:00 | 296.95 | 2025-11-07 11:15:00 | 284.75 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-10-28 09:30:00 | 297.20 | 2025-11-07 11:15:00 | 284.75 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2025-10-30 12:00:00 | 294.80 | 2025-11-07 11:15:00 | 284.75 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-10-30 13:15:00 | 295.00 | 2025-11-07 11:15:00 | 284.75 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-10-30 13:45:00 | 295.00 | 2025-11-07 14:15:00 | 280.06 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-10-31 09:15:00 | 294.10 | 2025-11-07 15:15:00 | 279.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 295.00 | 2025-11-10 09:15:00 | 284.45 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-10-31 09:15:00 | 294.10 | 2025-11-10 09:15:00 | 284.45 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-11-04 11:30:00 | 289.80 | 2025-11-12 12:15:00 | 284.40 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2025-11-04 14:00:00 | 289.35 | 2025-11-12 12:15:00 | 284.40 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2025-11-17 10:15:00 | 280.75 | 2025-11-24 09:15:00 | 266.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:45:00 | 280.60 | 2025-11-24 09:15:00 | 266.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 10:15:00 | 280.75 | 2025-11-24 14:15:00 | 269.70 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2025-11-18 09:45:00 | 280.60 | 2025-11-24 14:15:00 | 269.70 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest1 | 2025-12-05 09:30:00 | 266.50 | 2025-12-05 15:15:00 | 270.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-08 09:15:00 | 267.80 | 2025-12-10 14:15:00 | 266.65 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-12-31 12:45:00 | 285.45 | 2025-12-31 13:15:00 | 283.05 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-13 11:30:00 | 265.30 | 2026-01-19 09:15:00 | 273.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-01-28 14:30:00 | 261.25 | 2026-02-01 10:15:00 | 263.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-02-19 11:00:00 | 256.85 | 2026-02-23 14:15:00 | 260.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-23 13:00:00 | 257.05 | 2026-02-23 14:15:00 | 260.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-02-23 14:15:00 | 257.00 | 2026-02-23 14:15:00 | 260.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-03-09 14:30:00 | 262.95 | 2026-03-11 13:15:00 | 259.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-12 14:15:00 | 259.65 | 2026-03-16 10:15:00 | 246.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 259.65 | 2026-03-16 15:15:00 | 251.25 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2026-03-13 09:45:00 | 258.20 | 2026-03-18 10:15:00 | 259.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-02 09:15:00 | 240.33 | 2026-04-08 10:15:00 | 251.34 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2026-04-20 12:15:00 | 276.12 | 2026-04-23 14:15:00 | 274.71 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-05-05 10:45:00 | 272.35 | 2026-05-07 09:15:00 | 277.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-05-05 14:00:00 | 272.15 | 2026-05-07 09:15:00 | 277.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-05-05 15:00:00 | 272.40 | 2026-05-07 09:15:00 | 277.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-05-06 12:15:00 | 271.55 | 2026-05-07 09:15:00 | 277.50 | STOP_HIT | 1.00 | -2.19% |
