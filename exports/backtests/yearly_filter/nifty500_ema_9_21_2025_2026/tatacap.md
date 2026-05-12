# Tata Capital Ltd. (TATACAP)

## Backtest Summary

- **Window:** 2025-10-13 09:15:00 → 2026-05-11 15:15:00 (982 bars)
- **Last close:** 314.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 53 |
| ALERT1 | 35 |
| ALERT2 | 34 |
| ALERT2_SKIP | 22 |
| ALERT3 | 93 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 27 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 22
- **Target hits / Stop hits / Partials:** 0 / 31 / 1
- **Avg / median % per leg:** -0.49% / -0.58%
- **Sum % (uncompounded):** -15.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 7 | 36.8% | 0 | 19 | 0 | -0.18% | -3.5% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.30% | 1.2% |
| BUY @ 3rd Alert (retest2) | 15 | 3 | 20.0% | 0 | 15 | 0 | -0.31% | -4.7% |
| SELL (all) | 13 | 3 | 23.1% | 0 | 12 | 1 | -0.93% | -12.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 0 | 12 | 1 | -0.93% | -12.1% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.30% | 1.2% |
| retest2 (combined) | 28 | 6 | 21.4% | 0 | 27 | 1 | -0.60% | -16.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 326.00 | 329.54 | 329.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 325.75 | 328.78 | 329.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 330.30 | 327.33 | 328.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 330.30 | 327.33 | 328.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 330.30 | 327.33 | 328.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 330.30 | 327.33 | 328.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 329.00 | 327.66 | 328.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 327.30 | 327.66 | 328.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 329.00 | 326.75 | 326.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 329.00 | 326.75 | 326.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 330.60 | 328.53 | 327.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 326.80 | 329.46 | 328.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 326.80 | 329.46 | 328.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 326.80 | 329.46 | 328.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 326.35 | 329.46 | 328.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 326.05 | 328.78 | 328.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 326.00 | 328.78 | 328.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 326.30 | 327.82 | 328.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 325.35 | 326.72 | 327.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 13:15:00 | 327.15 | 326.58 | 327.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 327.15 | 326.58 | 327.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 327.15 | 326.58 | 327.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 327.15 | 326.58 | 327.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 326.90 | 326.65 | 327.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 325.60 | 326.59 | 326.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 328.15 | 327.09 | 327.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 328.15 | 327.09 | 327.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 10:15:00 | 330.15 | 328.81 | 328.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 327.55 | 328.77 | 328.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 13:15:00 | 327.55 | 328.77 | 328.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 327.55 | 328.77 | 328.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 327.55 | 328.77 | 328.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 328.25 | 328.66 | 328.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 330.00 | 328.66 | 328.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 325.40 | 328.22 | 328.15 | SL hit (close<static) qty=1.00 sl=327.30 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 325.80 | 327.74 | 327.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 324.40 | 326.35 | 327.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 325.10 | 323.59 | 324.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 325.10 | 323.59 | 324.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 325.10 | 323.59 | 324.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 325.85 | 323.59 | 324.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 325.05 | 323.88 | 324.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 325.05 | 323.88 | 324.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 325.10 | 324.13 | 324.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 325.10 | 324.13 | 324.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 325.00 | 324.30 | 324.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:45:00 | 325.15 | 324.30 | 324.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 324.55 | 324.35 | 324.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 324.00 | 324.35 | 324.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 326.00 | 324.72 | 324.87 | SL hit (close>static) qty=1.00 sl=325.25 alert=retest2 |

### Cycle 6 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 325.60 | 325.09 | 325.02 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 324.80 | 325.12 | 325.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 323.65 | 324.77 | 324.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 12:15:00 | 325.00 | 324.40 | 324.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 325.00 | 324.40 | 324.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 325.00 | 324.40 | 324.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 324.95 | 324.40 | 324.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 325.00 | 324.52 | 324.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 325.20 | 324.52 | 324.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 325.55 | 324.73 | 324.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 325.55 | 324.73 | 324.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 325.55 | 324.89 | 324.83 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 322.05 | 324.43 | 324.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 321.50 | 322.71 | 323.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 323.20 | 318.36 | 318.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 323.20 | 318.36 | 318.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 323.20 | 318.36 | 318.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 323.20 | 318.36 | 318.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 321.95 | 319.08 | 319.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 320.85 | 319.08 | 319.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 320.60 | 319.38 | 319.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 320.60 | 319.38 | 319.33 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 318.90 | 319.29 | 319.31 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 320.45 | 319.53 | 319.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 325.00 | 320.62 | 319.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 324.15 | 324.49 | 323.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:15:00 | 325.15 | 324.49 | 323.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 11:30:00 | 324.95 | 324.44 | 323.64 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 12:30:00 | 324.95 | 324.50 | 323.74 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 13:30:00 | 325.00 | 324.53 | 323.82 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 327.80 | 325.16 | 324.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:45:00 | 326.55 | 325.16 | 324.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 326.00 | 327.09 | 326.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 326.00 | 327.09 | 326.03 | SL hit (close<ema400) qty=1.00 sl=326.03 alert=retest1 |

### Cycle 13 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 323.35 | 325.19 | 325.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 321.50 | 323.50 | 324.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 323.10 | 322.09 | 322.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 323.10 | 322.09 | 322.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 323.10 | 322.09 | 322.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 323.10 | 322.09 | 322.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 322.55 | 322.18 | 322.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 323.00 | 322.18 | 322.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 322.60 | 322.26 | 322.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 322.50 | 322.26 | 322.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 322.70 | 322.35 | 322.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 322.70 | 322.35 | 322.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 322.45 | 322.41 | 322.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 322.50 | 322.41 | 322.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 321.75 | 322.24 | 322.65 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 326.60 | 323.35 | 323.02 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 323.50 | 323.71 | 323.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 321.40 | 323.25 | 323.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 324.20 | 323.44 | 323.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 324.20 | 323.44 | 323.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 324.20 | 323.44 | 323.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 324.20 | 323.44 | 323.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 326.00 | 323.95 | 323.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 15:15:00 | 327.00 | 325.46 | 324.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 326.15 | 326.36 | 325.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:45:00 | 326.45 | 326.36 | 325.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 325.45 | 326.15 | 325.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:30:00 | 325.10 | 326.15 | 325.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 324.65 | 325.85 | 325.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 325.40 | 325.85 | 325.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 325.60 | 325.80 | 325.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 326.35 | 325.74 | 325.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 326.40 | 325.90 | 325.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 324.50 | 325.39 | 325.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 324.50 | 325.39 | 325.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 322.20 | 324.75 | 325.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 325.25 | 323.64 | 324.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 325.25 | 323.64 | 324.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 325.25 | 323.64 | 324.21 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 326.50 | 324.81 | 324.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 09:15:00 | 327.10 | 325.75 | 325.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 15:15:00 | 325.60 | 326.13 | 325.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 325.60 | 326.13 | 325.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 325.60 | 326.13 | 325.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 324.30 | 326.13 | 325.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 323.75 | 325.65 | 325.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 323.75 | 325.65 | 325.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 322.75 | 325.07 | 325.24 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 328.00 | 324.71 | 324.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 342.05 | 331.08 | 328.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 344.55 | 345.70 | 339.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:45:00 | 344.85 | 345.70 | 339.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 338.60 | 342.94 | 340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 338.60 | 342.94 | 340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 339.20 | 342.19 | 340.45 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 335.45 | 339.33 | 339.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 334.55 | 338.37 | 339.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 337.60 | 336.58 | 337.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 337.60 | 336.58 | 337.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 337.60 | 336.58 | 337.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 337.60 | 336.58 | 337.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 338.05 | 336.87 | 337.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 335.75 | 336.87 | 337.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 336.50 | 336.80 | 337.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 340.80 | 337.31 | 337.42 | SL hit (close>static) qty=1.00 sl=338.35 alert=retest2 |

### Cycle 22 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 340.65 | 337.98 | 337.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 343.70 | 339.13 | 338.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 341.10 | 341.75 | 340.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:00:00 | 341.10 | 341.75 | 340.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 340.00 | 341.56 | 340.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 340.00 | 341.56 | 340.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 341.50 | 341.55 | 340.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 343.95 | 341.55 | 340.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 353.95 | 357.19 | 357.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 353.95 | 357.19 | 357.31 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 359.10 | 357.52 | 357.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 11:15:00 | 360.50 | 358.50 | 357.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 352.70 | 358.36 | 358.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 352.70 | 358.36 | 358.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 352.70 | 358.36 | 358.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 351.25 | 358.36 | 358.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 352.40 | 357.17 | 357.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 14:15:00 | 347.05 | 353.47 | 355.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 354.35 | 353.48 | 355.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 354.35 | 353.48 | 355.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 354.35 | 353.48 | 355.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 354.90 | 353.48 | 355.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 354.50 | 353.69 | 355.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 356.30 | 353.69 | 355.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 353.75 | 353.70 | 354.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 354.50 | 353.70 | 354.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 353.00 | 352.05 | 353.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 353.00 | 352.05 | 353.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 353.60 | 352.36 | 353.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 356.00 | 352.36 | 353.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 354.70 | 352.83 | 353.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 354.70 | 352.83 | 353.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 355.90 | 353.44 | 353.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 355.90 | 353.44 | 353.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 356.10 | 353.97 | 354.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 357.40 | 353.97 | 354.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 355.50 | 354.32 | 354.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 357.90 | 355.04 | 354.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 359.20 | 359.37 | 358.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 10:15:00 | 359.20 | 359.37 | 358.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 359.20 | 359.37 | 358.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 357.35 | 359.37 | 358.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 358.85 | 359.46 | 358.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 358.85 | 359.46 | 358.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 355.60 | 358.68 | 358.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 355.60 | 358.68 | 358.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 360.80 | 359.11 | 358.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:45:00 | 358.85 | 359.11 | 358.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 357.20 | 358.73 | 358.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 352.10 | 358.73 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 353.85 | 357.75 | 357.92 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 359.20 | 357.97 | 357.88 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 357.00 | 357.95 | 357.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 09:15:00 | 353.30 | 357.02 | 357.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 333.80 | 333.26 | 337.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 333.80 | 333.26 | 337.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 338.85 | 334.25 | 337.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 338.85 | 334.25 | 337.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 337.05 | 334.81 | 337.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 338.85 | 334.81 | 337.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 339.90 | 335.83 | 337.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 339.90 | 335.83 | 337.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 341.45 | 336.95 | 337.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 341.45 | 336.95 | 337.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 339.40 | 338.22 | 338.17 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 334.85 | 337.60 | 337.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 332.20 | 336.52 | 337.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 340.50 | 333.72 | 334.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 340.50 | 333.72 | 334.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 340.50 | 333.72 | 334.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:00:00 | 340.50 | 333.72 | 334.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 344.95 | 337.36 | 336.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 346.30 | 341.51 | 338.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 344.50 | 346.31 | 343.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 344.50 | 346.31 | 343.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 344.50 | 346.31 | 343.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 344.50 | 346.31 | 343.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 343.00 | 345.35 | 343.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 343.00 | 345.35 | 343.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 346.50 | 345.58 | 343.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 346.75 | 345.60 | 344.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 346.80 | 345.67 | 344.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 341.70 | 344.57 | 344.05 | SL hit (close<static) qty=1.00 sl=342.15 alert=retest2 |

### Cycle 33 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 352.50 | 354.38 | 354.60 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 357.25 | 354.73 | 354.56 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 353.90 | 355.17 | 355.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 352.10 | 354.56 | 354.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 339.80 | 338.60 | 341.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 339.80 | 338.60 | 341.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 339.80 | 338.60 | 341.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 336.60 | 338.20 | 340.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 319.77 | 332.64 | 334.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 327.00 | 323.34 | 326.44 | SL hit (close>ema200) qty=0.50 sl=323.34 alert=retest2 |

### Cycle 36 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 322.05 | 318.77 | 318.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 322.50 | 320.01 | 319.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 319.70 | 319.95 | 319.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 319.70 | 319.95 | 319.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 319.70 | 319.95 | 319.26 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 314.70 | 318.70 | 318.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 312.65 | 315.61 | 316.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 309.10 | 308.60 | 311.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 309.30 | 308.60 | 311.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 310.90 | 309.36 | 311.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 310.90 | 309.36 | 311.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 311.45 | 309.78 | 311.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 311.45 | 309.78 | 311.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 311.00 | 310.02 | 311.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 310.90 | 310.02 | 311.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 312.30 | 310.48 | 311.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 312.30 | 310.48 | 311.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 310.90 | 310.56 | 311.40 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 316.70 | 312.26 | 311.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 317.60 | 313.33 | 312.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 318.50 | 318.67 | 316.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 318.50 | 318.67 | 316.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 315.50 | 317.68 | 316.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 315.50 | 317.68 | 316.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 312.55 | 316.66 | 315.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 312.40 | 316.66 | 315.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 312.90 | 315.24 | 315.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 310.80 | 314.35 | 314.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 314.30 | 313.73 | 314.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 14:00:00 | 314.30 | 313.73 | 314.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 317.25 | 314.44 | 314.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 317.25 | 314.44 | 314.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 316.90 | 314.93 | 314.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 311.50 | 314.93 | 314.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 315.75 | 313.30 | 313.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 318.80 | 314.85 | 314.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 318.80 | 314.85 | 314.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 322.60 | 316.40 | 315.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 11:15:00 | 321.80 | 321.94 | 319.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 12:00:00 | 321.80 | 321.94 | 319.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 315.65 | 321.82 | 320.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 315.65 | 321.82 | 320.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 316.75 | 320.80 | 320.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 317.45 | 320.31 | 319.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 315.50 | 319.82 | 319.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 315.50 | 319.82 | 319.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 312.10 | 316.56 | 318.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 11:15:00 | 307.30 | 307.01 | 310.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 12:00:00 | 307.30 | 307.01 | 310.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 310.00 | 307.82 | 309.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:45:00 | 309.50 | 307.82 | 309.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 309.60 | 308.18 | 309.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 306.60 | 307.93 | 309.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:30:00 | 306.05 | 307.60 | 308.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 323.00 | 311.23 | 309.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 323.00 | 311.23 | 309.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 323.55 | 313.69 | 311.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 11:15:00 | 326.35 | 326.78 | 323.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:45:00 | 326.30 | 326.78 | 323.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 324.05 | 326.21 | 324.26 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 320.90 | 323.49 | 323.52 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 328.50 | 324.23 | 323.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 330.50 | 326.48 | 325.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 335.55 | 336.21 | 334.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 335.55 | 336.21 | 334.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 335.55 | 336.21 | 334.10 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 333.25 | 334.06 | 334.12 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 335.00 | 334.26 | 334.20 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 332.65 | 334.02 | 334.13 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 337.40 | 334.54 | 334.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 340.95 | 336.35 | 335.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 15:15:00 | 339.00 | 339.28 | 337.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 09:15:00 | 338.35 | 339.28 | 337.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 339.25 | 339.27 | 337.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 341.20 | 339.27 | 337.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 338.35 | 339.09 | 337.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:45:00 | 339.05 | 339.09 | 337.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 336.90 | 338.65 | 337.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 336.90 | 338.65 | 337.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 337.90 | 338.50 | 337.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 338.25 | 338.17 | 337.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 335.00 | 337.54 | 337.40 | SL hit (close<static) qty=1.00 sl=336.20 alert=retest2 |

### Cycle 49 — SELL (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 09:15:00 | 334.30 | 336.89 | 337.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 11:15:00 | 332.95 | 335.77 | 336.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 329.80 | 328.55 | 330.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 329.80 | 328.55 | 330.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 329.80 | 328.55 | 330.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 331.00 | 328.55 | 330.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 327.75 | 326.92 | 329.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:45:00 | 329.70 | 326.92 | 329.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 328.30 | 327.20 | 328.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 328.65 | 327.20 | 328.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 330.95 | 327.95 | 329.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 330.95 | 327.95 | 329.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 332.20 | 328.80 | 329.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:15:00 | 333.70 | 328.80 | 329.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 334.65 | 329.97 | 329.88 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 327.95 | 329.86 | 330.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 325.85 | 328.42 | 329.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 326.50 | 325.99 | 327.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 326.50 | 325.99 | 327.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 326.50 | 325.99 | 327.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 324.75 | 325.80 | 326.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 328.55 | 326.41 | 326.97 | SL hit (close>static) qty=1.00 sl=327.90 alert=retest2 |

### Cycle 52 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 329.05 | 327.52 | 327.39 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 15:15:00 | 326.00 | 327.28 | 327.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 323.15 | 326.45 | 326.99 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-23 09:15:00 | 327.30 | 2025-10-27 12:15:00 | 329.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-10-31 11:00:00 | 325.60 | 2025-11-03 09:15:00 | 328.15 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-04 15:15:00 | 330.00 | 2025-11-06 09:15:00 | 325.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-10 15:15:00 | 324.00 | 2025-11-11 10:15:00 | 326.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-24 11:15:00 | 320.85 | 2025-11-24 11:15:00 | 320.60 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest1 | 2025-11-27 09:15:00 | 325.15 | 2025-12-01 09:15:00 | 326.00 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest1 | 2025-11-27 11:30:00 | 324.95 | 2025-12-01 09:15:00 | 326.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2025-11-27 12:30:00 | 324.95 | 2025-12-01 09:15:00 | 326.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2025-11-27 13:30:00 | 325.00 | 2025-12-01 09:15:00 | 326.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-12-11 14:00:00 | 326.35 | 2025-12-12 15:15:00 | 324.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-12 09:30:00 | 326.40 | 2025-12-12 15:15:00 | 324.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-30 09:15:00 | 335.75 | 2025-12-31 09:15:00 | 340.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-12-30 10:00:00 | 336.50 | 2025-12-31 09:15:00 | 340.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-02 09:15:00 | 343.95 | 2026-01-08 14:15:00 | 353.95 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-05 13:45:00 | 346.75 | 2026-02-06 10:15:00 | 341.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-02-05 15:15:00 | 346.80 | 2026-02-06 10:15:00 | 341.70 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-02-06 15:00:00 | 346.60 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2026-02-09 09:15:00 | 348.85 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2026-02-10 11:00:00 | 353.90 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-02-10 11:45:00 | 353.65 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-02-12 11:45:00 | 355.25 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-02-13 09:45:00 | 353.95 | 2026-02-13 15:15:00 | 352.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-02-25 11:00:00 | 336.60 | 2026-03-02 09:15:00 | 319.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:00:00 | 336.60 | 2026-03-05 09:15:00 | 327.00 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-03-23 09:15:00 | 311.50 | 2026-03-24 11:15:00 | 318.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-03-24 10:00:00 | 315.75 | 2026-03-24 11:15:00 | 318.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-03-27 11:30:00 | 317.45 | 2026-03-30 09:15:00 | 315.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-04-06 11:00:00 | 306.60 | 2026-04-08 09:15:00 | 323.00 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-04-06 12:30:00 | 306.05 | 2026-04-08 09:15:00 | 323.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2026-04-24 15:00:00 | 338.25 | 2026-04-24 15:15:00 | 335.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-27 09:15:00 | 339.05 | 2026-04-27 09:15:00 | 334.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-06 12:15:00 | 324.75 | 2026-05-06 14:15:00 | 328.55 | STOP_HIT | 1.00 | -1.17% |
