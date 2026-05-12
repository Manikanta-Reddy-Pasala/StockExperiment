# POWERGRID (POWERGRID)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 94 |
| ALERT1 | 59 |
| ALERT2 | 58 |
| ALERT2_SKIP | 31 |
| ALERT3 | 159 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 53 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 47
- **Target hits / Stop hits / Partials:** 0 / 60 / 1
- **Avg / median % per leg:** -0.24% / -0.60%
- **Sum % (uncompounded):** -14.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 4 | 12.9% | 0 | 31 | 0 | -0.42% | -13.0% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.59% | -4.1% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 0 | 24 | 0 | -0.37% | -8.9% |
| SELL (all) | 30 | 10 | 33.3% | 0 | 29 | 1 | -0.04% | -1.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 10 | 33.3% | 0 | 29 | 1 | -0.04% | -1.3% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.59% | -4.1% |
| retest2 (combined) | 54 | 14 | 25.9% | 0 | 53 | 1 | -0.19% | -10.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 309.20 | 306.07 | 305.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 309.90 | 307.32 | 306.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 303.25 | 306.68 | 306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 302.90 | 305.92 | 306.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 299.40 | 304.00 | 305.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 12:15:00 | 295.25 | 294.68 | 297.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 13:00:00 | 295.25 | 294.68 | 297.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 299.30 | 295.61 | 297.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 299.30 | 295.61 | 297.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 299.45 | 296.37 | 298.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 298.25 | 296.99 | 298.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 298.35 | 298.23 | 298.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 299.80 | 298.65 | 298.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 299.80 | 298.65 | 298.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 300.30 | 298.98 | 298.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 298.10 | 300.76 | 300.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 296.05 | 298.68 | 299.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 292.00 | 291.86 | 294.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:15:00 | 294.10 | 291.86 | 294.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 295.30 | 292.55 | 294.80 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 298.05 | 296.19 | 295.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 301.40 | 297.53 | 296.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 297.40 | 298.43 | 297.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 296.75 | 298.10 | 297.41 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 293.25 | 296.93 | 296.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 292.30 | 293.74 | 294.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 292.95 | 292.84 | 294.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:00:00 | 292.95 | 292.84 | 294.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 293.20 | 292.78 | 293.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 293.20 | 292.78 | 293.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 291.40 | 292.49 | 293.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 292.55 | 292.49 | 293.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 289.60 | 291.69 | 292.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 293.00 | 291.69 | 292.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 292.15 | 291.42 | 292.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 292.00 | 291.42 | 292.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 292.05 | 291.55 | 292.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 292.25 | 291.55 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 292.50 | 291.74 | 292.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 292.85 | 291.74 | 292.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 292.15 | 291.82 | 292.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 291.40 | 292.14 | 292.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 290.90 | 291.69 | 292.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 293.80 | 290.56 | 290.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 293.80 | 290.56 | 290.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 295.00 | 291.44 | 290.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 12:15:00 | 294.15 | 294.23 | 292.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 13:00:00 | 294.15 | 294.23 | 292.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 298.00 | 299.53 | 298.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 298.00 | 299.53 | 298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 295.90 | 298.81 | 298.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 295.90 | 298.81 | 298.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 295.50 | 297.55 | 297.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 293.10 | 296.33 | 297.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 285.85 | 287.46 | 288.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 285.85 | 287.32 | 287.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 286.25 | 287.05 | 287.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 286.20 | 286.94 | 287.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 286.50 | 286.65 | 287.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 290.45 | 287.31 | 287.48 | SL hit (close>static) qty=1.00 sl=290.30 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 291.70 | 288.19 | 287.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 292.65 | 290.16 | 288.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 289.30 | 290.73 | 289.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 290.35 | 290.65 | 289.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 290.85 | 290.65 | 289.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 291.05 | 290.61 | 289.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 292.05 | 290.97 | 290.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:45:00 | 290.55 | 291.11 | 290.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 289.60 | 290.81 | 290.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 289.60 | 290.81 | 290.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 288.40 | 290.33 | 290.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 288.40 | 290.33 | 290.18 | SL hit (close<static) qty=1.00 sl=289.15 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 286.55 | 289.57 | 289.85 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 290.35 | 289.42 | 289.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 290.80 | 289.69 | 289.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 289.90 | 289.95 | 289.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 289.90 | 289.95 | 289.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 292.65 | 290.49 | 289.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:00:00 | 293.95 | 291.18 | 290.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 295.40 | 296.81 | 296.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 295.40 | 296.81 | 296.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 293.85 | 294.82 | 295.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 294.40 | 294.26 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 295.15 | 294.47 | 294.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 295.65 | 294.47 | 294.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 296.00 | 294.78 | 294.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 296.00 | 294.78 | 294.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 295.90 | 295.00 | 295.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 295.50 | 295.00 | 295.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 296.00 | 295.12 | 295.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 296.00 | 295.12 | 295.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 296.75 | 295.50 | 295.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 298.75 | 299.38 | 298.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 298.75 | 299.38 | 298.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 298.30 | 299.16 | 298.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 298.30 | 299.16 | 298.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 299.45 | 299.22 | 298.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 300.65 | 299.17 | 298.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 297.40 | 298.79 | 298.46 | SL hit (close<static) qty=1.00 sl=298.20 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 297.85 | 298.47 | 298.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 297.40 | 298.31 | 298.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 298.80 | 296.94 | 297.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 295.30 | 295.19 | 296.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 295.10 | 295.19 | 296.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 296.30 | 295.41 | 296.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 296.30 | 295.41 | 296.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 297.00 | 295.73 | 296.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 296.80 | 295.73 | 296.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 296.40 | 295.86 | 296.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 295.25 | 296.33 | 296.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 297.15 | 296.35 | 296.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 297.15 | 296.35 | 296.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 297.75 | 296.63 | 296.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 297.00 | 297.12 | 296.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 298.70 | 297.34 | 296.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:45:00 | 298.80 | 297.71 | 297.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 14:00:00 | 298.55 | 298.17 | 297.50 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 297.55 | 298.51 | 297.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 297.55 | 298.51 | 297.98 | SL hit (close<ema400) qty=1.00 sl=297.98 alert=retest1 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 294.35 | 297.90 | 297.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 294.00 | 297.12 | 297.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 294.40 | 293.97 | 295.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 294.40 | 293.97 | 295.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 293.05 | 292.76 | 293.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 293.50 | 292.76 | 293.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 293.00 | 292.86 | 293.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 293.45 | 292.86 | 293.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 294.75 | 293.24 | 293.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 294.75 | 293.24 | 293.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 295.60 | 293.71 | 293.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 295.60 | 293.71 | 293.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 295.50 | 294.07 | 294.00 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 289.35 | 293.42 | 293.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 286.50 | 288.40 | 289.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 286.00 | 285.20 | 286.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 285.85 | 285.33 | 286.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 285.70 | 285.33 | 286.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 284.50 | 285.17 | 285.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 283.85 | 284.71 | 285.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 286.80 | 285.20 | 285.22 | SL hit (close>static) qty=1.00 sl=286.25 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 286.05 | 285.37 | 285.29 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 284.90 | 285.27 | 285.29 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 287.15 | 285.65 | 285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 288.50 | 286.22 | 285.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 288.65 | 288.74 | 287.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 09:15:00 | 289.05 | 288.74 | 287.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 289.50 | 290.01 | 289.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 288.95 | 290.01 | 289.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 289.15 | 289.84 | 289.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 289.15 | 289.84 | 289.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 287.80 | 289.43 | 289.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 287.80 | 289.43 | 289.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 288.00 | 289.14 | 288.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 288.15 | 289.14 | 288.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 287.30 | 288.78 | 288.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 286.30 | 287.69 | 288.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 283.95 | 283.86 | 285.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:00:00 | 283.95 | 283.86 | 285.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 284.15 | 283.70 | 284.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 284.50 | 283.70 | 284.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 284.15 | 283.79 | 284.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 284.45 | 283.79 | 284.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 278.70 | 276.23 | 277.42 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 280.00 | 278.11 | 278.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 285.75 | 280.31 | 279.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 286.00 | 286.05 | 284.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:30:00 | 286.00 | 286.05 | 284.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 284.30 | 285.66 | 284.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 284.30 | 285.66 | 284.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 284.75 | 285.47 | 284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:15:00 | 284.15 | 285.47 | 284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 283.25 | 285.03 | 284.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 283.25 | 285.03 | 284.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 282.20 | 284.46 | 284.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 282.20 | 284.46 | 284.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 281.55 | 283.40 | 283.65 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 284.75 | 283.82 | 283.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 15:15:00 | 285.55 | 284.64 | 284.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 284.20 | 284.71 | 284.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 284.00 | 284.57 | 284.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 284.00 | 284.57 | 284.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 283.90 | 284.44 | 284.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 284.30 | 284.44 | 284.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 284.60 | 284.47 | 284.29 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 282.75 | 284.12 | 284.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 282.15 | 283.53 | 283.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 283.75 | 283.57 | 283.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:00:00 | 283.75 | 283.57 | 283.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 283.90 | 283.64 | 283.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 284.05 | 283.64 | 283.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 282.90 | 283.49 | 283.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 282.70 | 283.49 | 283.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 284.00 | 283.58 | 283.76 | SL hit (close>static) qty=1.00 sl=283.90 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 285.85 | 284.10 | 283.97 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 283.00 | 283.77 | 283.88 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 284.90 | 284.10 | 284.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 286.60 | 284.60 | 284.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 285.85 | 286.23 | 285.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 285.85 | 286.23 | 285.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 285.75 | 286.13 | 285.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 285.45 | 286.13 | 285.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 286.75 | 286.97 | 286.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 287.10 | 286.97 | 286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 285.90 | 286.76 | 286.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 285.90 | 286.76 | 286.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 286.10 | 286.63 | 286.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 287.30 | 286.64 | 286.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 287.35 | 287.61 | 287.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 287.10 | 287.51 | 287.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 286.20 | 287.06 | 287.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 286.20 | 287.06 | 287.08 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 287.35 | 287.12 | 287.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 287.50 | 287.20 | 287.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 287.15 | 287.33 | 287.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 286.55 | 287.18 | 287.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 286.35 | 287.18 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 287.20 | 287.18 | 287.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 287.85 | 287.47 | 287.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 286.95 | 287.32 | 287.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 286.95 | 287.32 | 287.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 286.10 | 287.07 | 287.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 287.05 | 286.92 | 287.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 286.70 | 286.88 | 287.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 287.15 | 286.88 | 287.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 286.95 | 286.89 | 287.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 287.15 | 286.89 | 287.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 286.95 | 286.90 | 287.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 287.00 | 286.90 | 287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 286.35 | 286.79 | 286.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 286.35 | 286.79 | 286.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 287.90 | 286.94 | 287.01 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 287.85 | 287.18 | 287.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 14:15:00 | 288.80 | 287.74 | 287.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 290.90 | 291.88 | 290.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 288.15 | 291.14 | 290.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 288.15 | 291.14 | 290.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 286.00 | 290.11 | 289.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 286.00 | 290.11 | 289.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 284.00 | 288.89 | 289.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 281.25 | 286.61 | 288.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 285.00 | 283.60 | 285.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 285.00 | 283.60 | 285.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 285.65 | 284.01 | 285.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 285.65 | 284.01 | 285.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 282.80 | 283.77 | 285.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:30:00 | 284.60 | 283.77 | 285.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 283.40 | 282.48 | 283.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 281.70 | 282.42 | 283.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 281.70 | 282.42 | 283.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 280.80 | 282.15 | 283.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 281.70 | 281.33 | 282.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 283.90 | 281.62 | 282.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 283.75 | 281.62 | 282.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 283.90 | 282.07 | 282.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 283.70 | 282.07 | 282.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 284.95 | 282.65 | 282.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 284.95 | 282.65 | 282.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 286.85 | 283.49 | 282.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 285.15 | 285.67 | 284.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 285.15 | 285.67 | 284.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 285.80 | 288.36 | 287.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 285.80 | 288.36 | 287.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 285.55 | 287.80 | 287.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 285.25 | 287.80 | 287.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 284.70 | 286.60 | 286.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 283.45 | 285.54 | 286.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 285.55 | 285.54 | 286.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 285.55 | 285.54 | 286.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 286.00 | 285.64 | 286.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 286.65 | 285.64 | 286.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 286.10 | 285.73 | 286.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 285.40 | 285.66 | 286.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 286.72 | 286.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 290.20 | 286.72 | 286.42 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 285.60 | 286.95 | 287.10 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 290.45 | 287.51 | 287.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 291.50 | 288.31 | 287.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 290.25 | 291.22 | 290.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 290.50 | 291.08 | 290.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 289.55 | 291.08 | 290.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 289.75 | 290.81 | 290.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 289.75 | 290.81 | 290.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 288.10 | 290.27 | 290.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 288.10 | 290.27 | 290.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 288.85 | 289.99 | 290.02 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 290.85 | 289.99 | 289.99 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 288.70 | 289.73 | 289.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 288.50 | 289.49 | 289.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 290.05 | 288.98 | 289.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 289.95 | 289.18 | 289.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 290.60 | 289.18 | 289.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 289.65 | 289.47 | 289.45 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 289.15 | 289.39 | 289.41 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 290.30 | 289.57 | 289.50 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 288.10 | 289.33 | 289.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 287.60 | 288.98 | 289.25 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 292.25 | 289.52 | 289.41 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 287.90 | 289.59 | 289.82 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 293.80 | 290.62 | 290.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 295.00 | 291.50 | 290.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 292.40 | 293.59 | 292.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 292.05 | 293.28 | 292.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 291.15 | 293.28 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 292.85 | 293.20 | 292.25 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 289.65 | 291.48 | 291.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 289.30 | 291.04 | 291.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 290.05 | 289.38 | 290.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 290.05 | 289.38 | 290.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 288.00 | 289.10 | 289.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 279.25 | 288.71 | 289.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 265.29 | 268.34 | 271.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 268.15 | 267.83 | 269.70 | SL hit (close>ema200) qty=0.50 sl=267.83 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 270.05 | 269.25 | 269.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 270.85 | 269.66 | 269.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 274.20 | 274.29 | 272.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 274.20 | 274.29 | 272.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 273.40 | 274.09 | 273.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 277.25 | 274.79 | 273.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 276.05 | 276.79 | 275.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 276.65 | 276.79 | 275.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 274.45 | 276.03 | 276.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 274.45 | 276.03 | 276.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 273.55 | 275.53 | 275.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 276.50 | 275.73 | 275.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 275.50 | 275.68 | 275.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 274.00 | 275.68 | 275.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 274.80 | 275.50 | 275.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:15:00 | 275.25 | 275.25 | 275.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 274.85 | 274.80 | 275.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 275.05 | 274.85 | 275.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 275.50 | 274.85 | 275.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 274.95 | 274.87 | 275.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 274.95 | 274.87 | 275.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 275.00 | 274.90 | 275.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 275.00 | 274.90 | 275.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 275.20 | 274.96 | 275.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 275.20 | 274.96 | 275.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 274.40 | 274.85 | 275.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 275.55 | 274.85 | 275.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 273.75 | 274.63 | 274.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 272.35 | 274.23 | 274.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 269.10 | 268.79 | 268.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 269.10 | 268.79 | 268.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 269.70 | 268.97 | 268.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 267.75 | 269.27 | 269.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 267.40 | 268.90 | 269.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 266.10 | 268.34 | 268.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 266.45 | 265.29 | 266.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 265.25 | 265.29 | 266.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 263.25 | 265.58 | 265.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 264.50 | 265.41 | 265.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 264.10 | 265.12 | 265.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 262.90 | 260.58 | 260.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 262.90 | 260.58 | 260.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 263.40 | 261.14 | 260.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 267.95 | 267.95 | 266.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 268.85 | 268.02 | 266.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:45:00 | 268.75 | 268.12 | 267.00 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:15:00 | 268.85 | 268.12 | 267.00 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:45:00 | 268.75 | 268.22 | 267.15 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 266.80 | 267.82 | 267.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 266.80 | 267.82 | 267.15 | SL hit (close<ema400) qty=1.00 sl=267.15 alert=retest1 |

### Cycle 56 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 262.80 | 266.13 | 266.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 262.70 | 265.44 | 266.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 263.80 | 261.26 | 262.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 264.50 | 261.91 | 262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 264.60 | 261.91 | 262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 264.30 | 263.12 | 263.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 264.90 | 263.48 | 263.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 266.65 | 268.40 | 268.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 264.80 | 267.33 | 268.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:00:00 | 259.70 | 260.87 | 262.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 260.50 | 258.84 | 258.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 260.50 | 258.84 | 258.81 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 256.80 | 258.60 | 258.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 256.05 | 257.47 | 257.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 256.50 | 255.79 | 256.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 256.15 | 255.86 | 256.64 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 259.40 | 257.16 | 256.91 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 254.60 | 256.81 | 256.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 253.75 | 255.77 | 256.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 254.75 | 254.68 | 255.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:30:00 | 255.15 | 254.68 | 255.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 255.00 | 254.28 | 254.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 257.50 | 254.28 | 254.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 257.65 | 254.95 | 255.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 257.65 | 254.95 | 255.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 259.40 | 255.84 | 255.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 260.85 | 258.83 | 257.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 256.95 | 259.42 | 258.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 256.10 | 258.76 | 258.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 256.10 | 258.76 | 258.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 254.50 | 257.91 | 257.99 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 262.00 | 257.86 | 257.83 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 254.60 | 257.66 | 257.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 252.20 | 256.57 | 257.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 253.95 | 253.90 | 255.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 254.85 | 253.90 | 255.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 256.25 | 254.17 | 255.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 256.25 | 254.17 | 255.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 262.25 | 255.78 | 256.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 262.25 | 255.78 | 256.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 266.90 | 258.01 | 257.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 270.25 | 260.46 | 258.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 287.10 | 287.16 | 280.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 287.10 | 287.16 | 280.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 288.45 | 290.29 | 288.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 288.45 | 290.29 | 288.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 289.35 | 290.10 | 288.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 289.65 | 290.10 | 288.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 15:00:00 | 289.50 | 289.98 | 288.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 291.30 | 289.82 | 288.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 289.85 | 292.32 | 292.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 289.85 | 292.32 | 292.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 11:15:00 | 289.35 | 291.72 | 292.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 294.40 | 290.33 | 291.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 298.80 | 292.03 | 291.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 299.75 | 294.53 | 292.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 11:15:00 | 297.50 | 297.78 | 295.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:30:00 | 297.40 | 297.78 | 295.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 298.20 | 299.80 | 298.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 298.20 | 299.80 | 298.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 297.55 | 299.35 | 298.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 298.15 | 299.35 | 298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 295.45 | 297.68 | 297.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 293.55 | 296.43 | 297.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 298.25 | 296.50 | 297.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 300.10 | 297.22 | 297.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 300.10 | 297.22 | 297.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 298.65 | 297.68 | 297.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 300.85 | 298.68 | 298.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 305.20 | 305.46 | 303.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 305.20 | 305.46 | 303.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 304.90 | 305.62 | 304.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 304.90 | 305.62 | 304.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 302.10 | 304.92 | 304.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 302.10 | 304.92 | 304.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 302.35 | 304.40 | 304.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:00:00 | 302.35 | 304.40 | 304.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 302.35 | 303.67 | 303.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 301.50 | 303.00 | 303.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 294.25 | 294.03 | 296.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 294.25 | 294.03 | 296.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 296.35 | 293.63 | 295.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 296.35 | 293.63 | 295.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 297.40 | 294.39 | 295.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 298.65 | 294.39 | 295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 297.15 | 295.45 | 296.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 298.55 | 295.45 | 296.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 298.90 | 296.38 | 296.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 298.90 | 296.38 | 296.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 299.00 | 296.91 | 296.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 301.15 | 297.76 | 297.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 299.15 | 299.76 | 298.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 14:45:00 | 299.00 | 299.76 | 298.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 300.00 | 299.81 | 298.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 291.80 | 299.81 | 298.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 289.75 | 297.80 | 297.86 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 299.30 | 296.64 | 296.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 300.05 | 298.52 | 297.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 298.50 | 299.43 | 298.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 298.70 | 299.28 | 298.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 297.55 | 299.28 | 298.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 300.00 | 299.43 | 298.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 301.80 | 299.43 | 298.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 13:15:00 | 300.50 | 302.38 | 301.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 300.65 | 302.04 | 301.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 301.20 | 301.83 | 301.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 301.00 | 301.66 | 301.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 299.25 | 301.66 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 297.15 | 300.76 | 300.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 297.15 | 300.76 | 300.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 291.90 | 298.99 | 299.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 299.30 | 297.48 | 298.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 296.90 | 297.38 | 298.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 294.70 | 297.37 | 298.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 302.25 | 298.09 | 298.19 | SL hit (close>static) qty=1.00 sl=302.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 302.80 | 299.04 | 298.61 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 298.20 | 298.67 | 298.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 296.50 | 298.24 | 298.52 | Break + close below crossover candle low |

### Cycle 79 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 304.10 | 299.30 | 298.95 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 294.10 | 298.69 | 299.12 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 301.55 | 299.45 | 299.30 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 294.00 | 298.45 | 298.88 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 300.15 | 298.98 | 298.90 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 13:15:00 | 297.05 | 298.69 | 298.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 294.95 | 297.94 | 298.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 297.15 | 295.73 | 296.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 296.55 | 295.89 | 296.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:30:00 | 296.60 | 295.89 | 296.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 295.50 | 295.82 | 296.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:45:00 | 295.70 | 295.82 | 296.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 295.50 | 295.73 | 296.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 297.00 | 295.73 | 296.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 297.15 | 296.00 | 296.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 297.15 | 296.00 | 296.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 296.30 | 296.06 | 296.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 294.85 | 295.84 | 296.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 295.00 | 292.09 | 291.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 295.00 | 292.09 | 291.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 296.50 | 294.52 | 293.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 12:15:00 | 294.75 | 295.32 | 294.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:45:00 | 294.85 | 295.32 | 294.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 294.50 | 295.15 | 294.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 294.30 | 295.15 | 294.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 294.80 | 295.08 | 294.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 297.30 | 295.07 | 294.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 316.50 | 318.90 | 319.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 316.50 | 318.90 | 319.00 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 323.10 | 318.37 | 318.27 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 316.30 | 319.22 | 319.57 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 321.70 | 319.27 | 319.23 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 13:15:00 | 318.00 | 319.15 | 319.21 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 319.65 | 319.30 | 319.27 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 318.30 | 319.10 | 319.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 316.45 | 318.57 | 318.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 320.50 | 318.55 | 318.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 320.70 | 318.98 | 319.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 320.70 | 318.98 | 319.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 319.70 | 319.13 | 319.07 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 318.50 | 319.00 | 319.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 316.90 | 318.58 | 318.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 315.05 | 313.90 | 314.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 09:15:00 | 298.25 | 2025-05-16 13:15:00 | 299.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-05-16 12:15:00 | 298.35 | 2025-05-16 13:15:00 | 299.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-06-03 09:30:00 | 291.40 | 2025-06-05 10:15:00 | 293.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-06-03 10:30:00 | 290.90 | 2025-06-05 10:15:00 | 293.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-18 12:45:00 | 285.85 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-19 09:15:00 | 285.85 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-19 10:30:00 | 286.25 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-19 12:15:00 | 286.20 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-23 11:15:00 | 290.85 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-23 12:15:00 | 291.05 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-24 09:15:00 | 292.05 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-24 10:45:00 | 290.55 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-26 15:00:00 | 293.95 | 2025-07-02 13:15:00 | 295.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-07-07 12:15:00 | 295.50 | 2025-07-07 14:15:00 | 296.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-07-11 09:15:00 | 300.65 | 2025-07-11 10:15:00 | 297.40 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-14 10:00:00 | 299.95 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-14 11:30:00 | 299.80 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-14 12:00:00 | 299.90 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-22 10:15:00 | 295.25 | 2025-07-22 11:15:00 | 297.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-07-23 10:15:00 | 298.70 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-23 10:45:00 | 298.80 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-07-23 14:00:00 | 298.55 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-08-11 09:30:00 | 283.85 | 2025-08-12 09:15:00 | 286.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-09 13:15:00 | 282.70 | 2025-09-09 14:15:00 | 284.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-09-16 09:15:00 | 287.30 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-17 10:30:00 | 287.35 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-09-17 11:30:00 | 287.10 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-18 14:30:00 | 287.85 | 2025-09-19 13:15:00 | 286.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-30 10:45:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-30 11:15:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-30 11:45:00 | 280.80 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-01 11:00:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-09 14:00:00 | 285.40 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-11-04 09:15:00 | 279.25 | 2025-11-11 10:15:00 | 265.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 279.25 | 2025-11-11 15:15:00 | 268.15 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-11-20 09:15:00 | 277.25 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-21 09:30:00 | 276.05 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-21 10:15:00 | 276.65 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-25 09:15:00 | 274.00 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2025-11-25 10:00:00 | 274.80 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-11-25 13:15:00 | 275.25 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-11-26 11:00:00 | 274.85 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-11-28 09:15:00 | 272.35 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-12-11 09:15:00 | 263.25 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-12-11 14:45:00 | 264.50 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-12-12 10:45:00 | 264.10 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest1 | 2025-12-26 09:15:00 | 268.85 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-12-26 10:45:00 | 268.75 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-12-26 11:15:00 | 268.85 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-12-26 11:45:00 | 268.75 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-09 12:00:00 | 259.70 | 2026-01-16 09:15:00 | 260.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-09 14:15:00 | 289.65 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2026-02-09 15:00:00 | 289.50 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-02-10 09:15:00 | 291.30 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-03-12 10:15:00 | 301.80 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-13 13:15:00 | 300.50 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-13 14:00:00 | 300.65 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-13 14:30:00 | 301.20 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-03-17 11:15:00 | 294.70 | 2026-03-18 09:15:00 | 302.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-04-01 11:00:00 | 294.85 | 2026-04-06 15:15:00 | 295.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-04-09 09:15:00 | 297.30 | 2026-04-24 09:15:00 | 316.50 | STOP_HIT | 1.00 | 6.46% |
