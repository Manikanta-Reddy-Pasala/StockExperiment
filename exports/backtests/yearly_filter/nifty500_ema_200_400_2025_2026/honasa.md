# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 354.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 14
- **Target hits / Stop hits / Partials:** 0 / 14 / 0
- **Avg / median % per leg:** -2.67% / -2.73%
- **Sum % (uncompounded):** -37.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.62% | -31.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.62% | -31.4% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.00% | -6.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.00% | -6.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.67% | -37.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 264.90 | 283.81 | 283.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 264.10 | 283.10 | 283.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 299.80 | 276.78 | 279.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 284.45 | 277.63 | 280.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 284.30 | 277.63 | 280.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 283.20 | 277.69 | 280.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 282.65 | 277.72 | 280.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:30:00 | 281.45 | 278.08 | 280.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 290.50 | 278.58 | 280.46 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 299.55 | 282.28 | 282.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 302.90 | 283.03 | 282.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 297.00 | 297.02 | 292.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 297.00 | 297.02 | 292.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 284.85 | 296.89 | 292.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 284.95 | 296.89 | 292.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 284.70 | 296.77 | 292.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 14:45:00 | 285.35 | 296.24 | 292.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 277.55 | 295.94 | 291.95 | SL hit (close<static) qty=1.00 sl=282.75 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 276.80 | 289.41 | 289.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 276.50 | 289.08 | 289.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 283.75 | 283.40 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 285.30 | 283.42 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 283.65 | 283.42 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 287.00 | 283.46 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 287.00 | 283.46 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 288.55 | 283.51 | 286.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 288.55 | 283.51 | 286.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 284.55 | 283.58 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 284.85 | 283.58 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 278.10 | 281.08 | 284.23 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 293.05 | 286.26 | 286.25 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 279.80 | 286.24 | 286.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 279.30 | 286.11 | 286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 272.00 | 271.78 | 277.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 272.00 | 271.78 | 277.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 276.25 | 271.96 | 277.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 275.85 | 271.96 | 277.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 275.80 | 271.90 | 276.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 275.80 | 271.90 | 276.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 276.20 | 271.95 | 276.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 271.10 | 271.95 | 276.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 283.90 | 272.06 | 276.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 283.90 | 272.06 | 276.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 286.70 | 272.21 | 276.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 288.40 | 272.21 | 276.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 13:15:00 | 296.00 | 280.42 | 280.39 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 271.50 | 280.87 | 280.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 267.35 | 280.39 | 280.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 283.20 | 277.70 | 279.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 278.30 | 277.71 | 279.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 280.85 | 277.71 | 279.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 275.55 | 277.42 | 278.93 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 292.65 | 280.01 | 279.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 295.50 | 280.59 | 280.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 291.10 | 293.84 | 288.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 291.10 | 293.84 | 288.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 289.25 | 293.74 | 288.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 290.00 | 293.54 | 288.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 294.20 | 293.40 | 288.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 289.70 | 293.37 | 288.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 280.35 | 293.46 | 288.86 | SL hit (close<static) qty=1.00 sl=283.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 280.80 | 285.77 | 285.79 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 289.50 | 285.82 | 285.81 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 281.55 | 285.76 | 285.78 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 291.25 | 285.81 | 285.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 293.30 | 285.88 | 285.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-14 15:00:00 | 289.40 | 2025-07-18 13:15:00 | 286.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-14 12:30:00 | 282.65 | 2025-08-20 10:15:00 | 290.50 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-08-19 09:30:00 | 281.45 | 2025-08-20 10:15:00 | 290.50 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-09-26 14:45:00 | 285.35 | 2025-09-29 09:15:00 | 277.55 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-10-01 13:45:00 | 285.80 | 2025-10-03 09:15:00 | 282.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-03 13:30:00 | 285.65 | 2025-10-14 09:15:00 | 280.30 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-08 12:15:00 | 285.50 | 2025-10-14 09:15:00 | 280.30 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-03-04 13:15:00 | 290.00 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-05 09:15:00 | 294.20 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2026-03-05 11:00:00 | 289.70 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-03-10 09:15:00 | 290.50 | 2026-03-11 15:15:00 | 286.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-03-11 09:15:00 | 292.15 | 2026-03-11 15:15:00 | 286.90 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-03-11 14:30:00 | 291.10 | 2026-03-12 12:15:00 | 281.40 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-18 14:30:00 | 294.65 | 2026-03-18 15:15:00 | 280.00 | STOP_HIT | 1.00 | -4.97% |
