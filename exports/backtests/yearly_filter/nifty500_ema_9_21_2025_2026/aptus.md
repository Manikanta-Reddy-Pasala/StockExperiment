# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 25 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 51 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 39
- **Target hits / Stop hits / Partials:** 3 / 53 / 11
- **Avg / median % per leg:** 0.83% / -0.83%
- **Sum % (uncompounded):** 55.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 5 | 19.2% | 0 | 26 | 0 | -1.18% | -30.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 5 | 19.2% | 0 | 26 | 0 | -1.18% | -30.7% |
| SELL (all) | 41 | 23 | 56.1% | 3 | 27 | 11 | 2.10% | 86.0% |
| SELL @ 2nd Alert (retest1) | 9 | 9 | 100.0% | 0 | 5 | 4 | 4.26% | 38.3% |
| SELL @ 3rd Alert (retest2) | 32 | 14 | 43.8% | 3 | 22 | 7 | 1.49% | 47.7% |
| retest1 (combined) | 9 | 9 | 100.0% | 0 | 5 | 4 | 4.26% | 38.3% |
| retest2 (combined) | 58 | 19 | 32.8% | 3 | 48 | 7 | 0.29% | 17.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 323.05 | 320.63 | 320.51 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 14:15:00 | 320.45 | 320.79 | 320.82 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 321.20 | 320.81 | 320.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 320.50 | 320.75 | 320.78 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 322.20 | 320.97 | 320.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 325.05 | 322.00 | 321.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 323.75 | 325.36 | 323.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 327.25 | 325.53 | 324.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:30:00 | 325.85 | 328.88 | 328.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 334.00 | 339.94 | 340.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 334.00 | 339.94 | 340.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 309.30 | 331.30 | 334.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 310.45 | 308.08 | 311.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 310.45 | 308.08 | 311.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 311.70 | 308.81 | 311.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 311.70 | 308.81 | 311.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 312.85 | 309.62 | 311.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 312.85 | 309.62 | 311.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 314.35 | 310.56 | 312.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:45:00 | 314.40 | 310.56 | 312.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 316.00 | 313.52 | 313.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 322.30 | 315.27 | 314.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 328.15 | 328.99 | 325.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 329.45 | 328.99 | 325.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 321.85 | 327.03 | 325.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 321.85 | 327.03 | 325.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 324.50 | 326.53 | 325.38 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 321.30 | 324.48 | 324.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 320.15 | 323.61 | 324.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 320.00 | 319.22 | 320.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 320.00 | 319.22 | 320.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 316.50 | 318.78 | 320.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 312.30 | 316.47 | 317.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 312.00 | 313.97 | 314.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 318.80 | 314.81 | 314.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 318.80 | 314.81 | 314.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 320.55 | 315.95 | 315.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 322.50 | 322.81 | 320.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:45:00 | 322.30 | 322.81 | 320.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 320.25 | 322.33 | 320.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 320.25 | 322.33 | 320.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 317.30 | 321.32 | 320.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 317.30 | 321.32 | 320.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 320.15 | 320.24 | 320.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:15:00 | 323.00 | 320.24 | 320.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 319.20 | 320.47 | 320.25 | SL hit (close<static) qty=1.00 sl=319.80 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 318.50 | 319.91 | 320.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 13:15:00 | 317.75 | 319.27 | 319.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 323.00 | 319.76 | 319.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:15:00 | 323.75 | 319.76 | 319.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 324.05 | 320.62 | 320.18 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 320.00 | 321.23 | 321.25 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 322.40 | 321.46 | 321.35 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 321.00 | 321.76 | 321.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 320.40 | 321.49 | 321.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 322.95 | 320.63 | 321.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 322.95 | 320.63 | 321.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 322.10 | 320.93 | 321.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 322.25 | 320.93 | 321.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 320.70 | 321.12 | 321.24 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 324.00 | 321.69 | 321.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 325.65 | 322.48 | 321.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 11:15:00 | 346.20 | 347.03 | 344.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:30:00 | 345.65 | 347.03 | 344.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 342.30 | 345.89 | 344.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 341.80 | 345.89 | 344.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 345.05 | 345.72 | 344.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 346.60 | 345.35 | 344.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 346.30 | 345.27 | 344.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:00:00 | 346.25 | 345.49 | 344.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:00:00 | 346.35 | 345.74 | 345.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 346.00 | 345.79 | 345.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 345.90 | 345.79 | 345.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 335.60 | 343.76 | 344.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 335.60 | 343.76 | 344.39 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 348.65 | 345.11 | 344.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 350.60 | 346.48 | 345.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 349.85 | 350.24 | 348.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 349.85 | 350.24 | 348.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 349.05 | 350.00 | 348.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 348.75 | 350.00 | 348.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 346.30 | 349.26 | 348.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 346.30 | 349.26 | 348.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 347.75 | 348.96 | 348.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:15:00 | 349.00 | 348.96 | 348.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 348.80 | 348.93 | 348.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 349.80 | 348.62 | 348.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:00:00 | 350.05 | 349.85 | 349.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 349.95 | 349.55 | 349.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 353.85 | 350.51 | 349.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 349.95 | 351.08 | 350.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 349.50 | 351.08 | 350.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 347.50 | 350.36 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 350.00 | 350.36 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 347.50 | 349.79 | 349.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 347.50 | 349.79 | 349.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 346.90 | 349.21 | 349.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 346.90 | 349.21 | 349.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 344.45 | 347.14 | 348.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 10:15:00 | 333.90 | 332.60 | 336.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 11:00:00 | 333.90 | 332.60 | 336.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 335.80 | 333.24 | 336.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 336.25 | 333.24 | 336.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 334.85 | 333.56 | 336.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 331.55 | 333.56 | 335.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 338.00 | 335.06 | 335.80 | SL hit (close>static) qty=1.00 sl=337.45 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 13:15:00 | 337.55 | 335.26 | 335.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 09:15:00 | 342.15 | 337.26 | 336.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 344.75 | 346.33 | 342.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 344.75 | 346.33 | 342.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 342.05 | 345.15 | 343.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 342.05 | 345.15 | 343.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 343.90 | 344.90 | 343.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:45:00 | 344.40 | 343.89 | 343.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 345.75 | 344.01 | 343.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 340.30 | 343.15 | 343.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 340.30 | 343.15 | 343.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 334.15 | 338.54 | 340.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 337.00 | 335.14 | 336.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 337.75 | 335.14 | 336.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 337.45 | 335.60 | 336.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 337.45 | 335.60 | 336.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 340.95 | 336.67 | 337.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 340.95 | 336.67 | 337.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 340.50 | 337.44 | 337.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 340.50 | 337.44 | 337.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 344.85 | 338.92 | 338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 345.95 | 341.65 | 339.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 358.70 | 359.06 | 356.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:30:00 | 359.10 | 359.06 | 356.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 355.35 | 358.32 | 356.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 355.35 | 358.32 | 356.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 355.00 | 357.66 | 355.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 354.85 | 357.66 | 355.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 356.50 | 356.84 | 355.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 356.15 | 356.84 | 355.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 355.50 | 356.57 | 355.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 355.50 | 356.57 | 355.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 357.35 | 356.73 | 355.96 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 355.45 | 355.75 | 355.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 352.30 | 354.82 | 355.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 319.50 | 319.45 | 323.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:30:00 | 318.50 | 319.45 | 323.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 323.85 | 319.45 | 321.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 322.40 | 319.45 | 321.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 326.00 | 320.76 | 321.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 328.50 | 320.76 | 321.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 327.00 | 322.66 | 322.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 332.45 | 325.49 | 323.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 328.00 | 331.28 | 328.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 328.00 | 331.28 | 328.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 330.15 | 331.06 | 328.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 333.50 | 331.06 | 328.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:45:00 | 331.95 | 331.79 | 329.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 339.30 | 340.72 | 340.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 339.30 | 340.72 | 340.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 338.35 | 339.99 | 340.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 335.85 | 334.73 | 336.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 338.05 | 334.73 | 336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 335.40 | 334.86 | 336.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 335.40 | 334.86 | 336.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 334.00 | 334.69 | 335.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 330.75 | 334.10 | 334.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 332.10 | 330.71 | 331.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 329.95 | 331.50 | 331.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 315.50 | 319.44 | 321.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 314.21 | 317.90 | 320.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 313.45 | 317.90 | 320.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 313.55 | 313.32 | 316.45 | SL hit (close>ema200) qty=0.50 sl=313.32 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 320.40 | 316.72 | 316.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 323.00 | 320.61 | 319.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 326.45 | 326.47 | 323.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:30:00 | 326.95 | 326.47 | 323.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 325.10 | 326.13 | 323.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 324.75 | 326.13 | 323.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 324.20 | 325.74 | 324.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 324.40 | 325.74 | 324.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 322.55 | 325.10 | 323.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 322.55 | 325.10 | 323.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 323.00 | 324.68 | 323.80 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 320.25 | 322.99 | 323.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 318.60 | 322.12 | 322.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 312.05 | 310.46 | 312.73 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 319.85 | 313.89 | 313.76 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 313.45 | 314.12 | 314.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 311.00 | 313.50 | 313.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 313.30 | 311.25 | 312.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 313.30 | 311.25 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 313.25 | 311.65 | 312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 312.80 | 311.65 | 312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 312.45 | 311.81 | 312.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 314.00 | 311.81 | 312.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 313.10 | 312.07 | 312.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 313.10 | 312.07 | 312.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 312.35 | 312.13 | 312.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:00:00 | 310.85 | 311.87 | 312.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 310.70 | 308.03 | 309.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 312.15 | 309.42 | 309.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 312.15 | 309.42 | 309.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 314.50 | 310.82 | 309.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 311.10 | 313.07 | 311.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 310.40 | 313.07 | 311.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 311.35 | 312.73 | 311.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 309.90 | 312.73 | 311.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 309.40 | 312.06 | 311.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 309.40 | 312.06 | 311.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 310.40 | 310.84 | 310.86 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 316.25 | 311.72 | 311.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 12:15:00 | 319.70 | 315.03 | 313.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 316.50 | 317.18 | 315.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 315.00 | 317.18 | 315.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 314.40 | 316.62 | 315.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 314.40 | 316.62 | 315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 312.95 | 315.89 | 314.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 312.95 | 315.89 | 314.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 314.85 | 314.84 | 314.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:15:00 | 313.10 | 314.84 | 314.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 311.80 | 314.23 | 314.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 309.55 | 312.01 | 313.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 314.35 | 311.48 | 312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 314.35 | 311.48 | 312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 317.25 | 312.63 | 312.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 317.25 | 312.63 | 312.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 316.45 | 313.39 | 313.12 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 311.75 | 312.80 | 312.91 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 313.50 | 312.88 | 312.86 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 307.25 | 313.05 | 313.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 12:15:00 | 302.80 | 311.00 | 312.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 294.05 | 293.93 | 298.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 294.05 | 293.93 | 298.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 287.40 | 285.05 | 286.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:30:00 | 288.50 | 285.05 | 286.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 287.50 | 285.54 | 286.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 286.30 | 285.54 | 286.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 288.80 | 286.56 | 286.73 | SL hit (close>static) qty=1.00 sl=287.90 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 290.40 | 287.33 | 287.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 293.30 | 290.81 | 289.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 11:15:00 | 291.55 | 292.61 | 291.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 11:45:00 | 291.80 | 292.61 | 291.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 290.90 | 292.26 | 291.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 290.40 | 292.26 | 291.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 291.30 | 292.07 | 291.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:30:00 | 290.95 | 292.07 | 291.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 288.40 | 291.34 | 290.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 288.40 | 291.34 | 290.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 288.50 | 290.77 | 290.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 287.55 | 290.77 | 290.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 287.80 | 290.18 | 290.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 284.55 | 287.76 | 288.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 279.70 | 279.45 | 282.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 11:00:00 | 279.70 | 279.45 | 282.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 285.30 | 280.52 | 281.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 285.30 | 280.52 | 281.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 286.00 | 281.62 | 282.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 283.70 | 281.62 | 282.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 285.85 | 282.65 | 282.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 285.85 | 282.65 | 282.29 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 281.45 | 282.07 | 282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 279.60 | 281.57 | 281.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 281.85 | 281.57 | 281.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 281.95 | 281.57 | 281.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 280.85 | 281.42 | 281.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 280.85 | 281.42 | 281.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 277.60 | 280.56 | 281.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 276.90 | 280.56 | 281.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 283.60 | 280.66 | 280.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 283.60 | 280.66 | 280.53 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 278.90 | 280.45 | 280.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 276.75 | 279.53 | 280.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 276.20 | 275.77 | 277.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 276.20 | 275.77 | 277.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 277.70 | 276.15 | 277.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 278.70 | 276.72 | 277.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 281.05 | 277.59 | 277.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 281.30 | 277.59 | 277.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 281.25 | 278.32 | 278.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 283.55 | 279.80 | 278.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 280.30 | 281.52 | 280.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 280.40 | 281.52 | 280.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 280.10 | 281.24 | 280.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:00:00 | 281.95 | 281.21 | 280.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 281.40 | 281.26 | 280.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 282.85 | 281.01 | 280.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 278.45 | 280.50 | 280.21 | SL hit (close<static) qty=1.00 sl=279.35 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 279.45 | 279.99 | 280.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 278.80 | 279.75 | 279.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 279.85 | 279.33 | 279.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:15:00 | 283.35 | 279.33 | 279.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 284.95 | 280.45 | 280.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 287.00 | 281.76 | 280.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 12:15:00 | 292.90 | 293.12 | 290.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 13:00:00 | 292.90 | 293.12 | 290.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 293.35 | 293.17 | 290.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:45:00 | 291.45 | 293.17 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 291.00 | 292.46 | 291.18 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 289.55 | 290.49 | 290.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 288.85 | 290.24 | 290.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 289.25 | 287.14 | 288.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 289.25 | 287.14 | 288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 288.20 | 287.36 | 288.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 288.20 | 287.36 | 288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 283.25 | 283.42 | 284.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 283.25 | 283.42 | 284.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 282.95 | 283.28 | 284.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 281.80 | 282.82 | 283.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:30:00 | 282.00 | 282.48 | 283.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 282.05 | 282.48 | 283.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 282.05 | 282.42 | 283.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 281.95 | 282.29 | 282.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 280.75 | 281.96 | 282.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 277.25 | 281.82 | 282.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 278.05 | 277.04 | 279.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 286.95 | 280.60 | 279.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 286.95 | 280.60 | 279.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 289.00 | 282.28 | 280.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 278.90 | 281.60 | 280.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 278.90 | 281.60 | 280.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 277.95 | 280.87 | 280.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 277.95 | 280.87 | 280.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 278.50 | 279.78 | 279.86 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 282.35 | 280.22 | 280.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 285.00 | 283.01 | 281.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 284.50 | 286.64 | 285.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 283.05 | 286.64 | 285.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 285.90 | 286.49 | 285.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 287.35 | 286.43 | 285.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 282.90 | 285.61 | 285.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 282.90 | 285.61 | 285.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 279.55 | 283.66 | 284.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 275.60 | 275.18 | 277.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:00:00 | 274.15 | 274.98 | 277.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:30:00 | 273.85 | 274.69 | 276.81 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 12:00:00 | 273.55 | 274.69 | 276.81 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 14:00:00 | 273.60 | 274.44 | 276.33 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 273.45 | 274.20 | 275.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 275.50 | 274.20 | 275.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 266.00 | 267.14 | 269.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 265.70 | 266.29 | 268.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 260.44 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 260.16 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 259.87 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 259.92 | 264.02 | 266.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 261.40 | 261.24 | 263.70 | SL hit (close>ema200) qty=0.50 sl=261.24 alert=retest1 |

### Cycle 51 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 266.90 | 264.36 | 264.23 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 262.40 | 264.10 | 264.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 261.20 | 263.52 | 263.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 263.45 | 262.47 | 263.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 263.45 | 262.47 | 263.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 260.95 | 262.17 | 263.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 262.50 | 262.17 | 263.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 265.00 | 262.73 | 263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:45:00 | 265.40 | 262.73 | 263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 264.85 | 263.16 | 263.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 264.90 | 263.16 | 263.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 266.90 | 264.16 | 263.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 268.95 | 265.12 | 264.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 269.70 | 270.00 | 267.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:30:00 | 269.45 | 270.00 | 267.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 273.05 | 270.57 | 269.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 273.95 | 271.25 | 269.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:00:00 | 274.35 | 272.22 | 270.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 266.90 | 270.31 | 270.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 266.90 | 270.31 | 270.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 261.35 | 268.52 | 269.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 266.75 | 264.95 | 267.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 266.75 | 264.95 | 267.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 266.75 | 265.31 | 267.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 276.00 | 265.31 | 267.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 278.15 | 267.88 | 268.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:15:00 | 279.30 | 267.88 | 268.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 280.15 | 270.33 | 269.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 280.95 | 275.00 | 271.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 278.50 | 280.13 | 276.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 278.50 | 280.13 | 276.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 276.80 | 279.47 | 276.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 270.50 | 279.47 | 276.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 272.45 | 278.06 | 276.49 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 270.20 | 275.50 | 275.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 268.95 | 273.48 | 274.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 262.85 | 262.48 | 266.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 262.65 | 262.48 | 266.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 266.05 | 264.05 | 266.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 266.05 | 264.05 | 266.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 266.55 | 264.55 | 266.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 265.25 | 264.55 | 266.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 267.20 | 265.08 | 266.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 268.20 | 265.08 | 266.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 260.40 | 264.15 | 265.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 258.80 | 261.18 | 263.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 245.86 | 253.22 | 256.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 250.80 | 250.64 | 253.60 | SL hit (close>ema200) qty=0.50 sl=250.64 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 247.35 | 243.03 | 242.53 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 241.80 | 244.21 | 244.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 231.43 | 238.50 | 241.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 230.99 | 229.95 | 232.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 227.50 | 229.84 | 232.17 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 228.16 | 229.56 | 231.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 219.97 | 229.37 | 231.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 226.95 | 223.74 | 226.28 | SL hit (close>ema400) qty=1.00 sl=226.28 alert=retest1 |

### Cycle 59 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 231.62 | 228.21 | 227.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 233.52 | 229.96 | 228.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 229.66 | 229.91 | 228.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:30:00 | 230.00 | 229.91 | 228.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 228.37 | 229.58 | 229.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 228.37 | 229.58 | 229.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 228.43 | 229.35 | 228.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 225.64 | 229.35 | 228.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 226.70 | 228.82 | 228.74 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 227.97 | 228.65 | 228.67 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 229.64 | 228.85 | 228.76 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 227.03 | 228.38 | 228.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 226.21 | 227.88 | 228.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 223.01 | 221.70 | 223.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 223.16 | 221.70 | 223.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 224.23 | 222.26 | 223.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 224.23 | 222.26 | 223.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 223.18 | 222.44 | 223.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:00:00 | 221.57 | 222.27 | 223.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:00:00 | 222.02 | 222.11 | 223.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:45:00 | 222.10 | 221.89 | 222.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.49 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.92 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 210.99 | 214.47 | 217.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 199.41 | 208.04 | 211.90 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 207.06 | 204.96 | 204.91 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 202.93 | 204.82 | 204.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 199.69 | 203.79 | 204.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 200.89 | 199.69 | 201.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:00:00 | 200.89 | 199.69 | 201.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 204.89 | 198.75 | 200.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 204.89 | 198.75 | 200.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 205.22 | 200.04 | 200.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 205.22 | 200.04 | 200.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 204.83 | 201.77 | 201.44 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 196.25 | 201.52 | 201.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 12:15:00 | 195.29 | 197.22 | 198.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 198.00 | 197.37 | 198.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 197.46 | 197.37 | 198.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 199.85 | 197.87 | 198.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 199.85 | 197.87 | 198.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 200.00 | 198.30 | 198.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 205.92 | 198.30 | 198.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 208.68 | 200.37 | 199.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 211.80 | 205.60 | 202.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 245.10 | 246.64 | 241.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 245.10 | 246.64 | 241.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 244.83 | 246.28 | 241.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 241.69 | 246.28 | 241.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 244.60 | 245.43 | 243.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 251.55 | 245.43 | 243.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 256.35 | 258.20 | 258.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 256.35 | 258.20 | 258.42 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 263.30 | 259.41 | 258.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 263.70 | 260.27 | 259.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 261.25 | 262.35 | 260.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:30:00 | 265.50 | 261.78 | 261.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 265.50 | 262.61 | 261.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 265.45 | 262.96 | 261.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 257.65 | 261.32 | 261.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 257.65 | 261.32 | 261.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 256.85 | 260.42 | 261.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 262.15 | 259.83 | 260.30 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 265.15 | 261.33 | 260.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 266.15 | 262.77 | 261.72 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 15:00:00 | 327.25 | 2025-05-29 09:15:00 | 334.00 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2025-05-20 14:30:00 | 325.85 | 2025-05-29 09:15:00 | 334.00 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-06-19 12:15:00 | 312.30 | 2025-06-24 09:15:00 | 318.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-23 09:15:00 | 312.00 | 2025-06-24 09:15:00 | 318.80 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-06-26 15:15:00 | 323.00 | 2025-06-27 09:15:00 | 319.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-15 09:15:00 | 346.60 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-07-15 13:00:00 | 346.30 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-16 12:00:00 | 346.25 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-07-16 14:00:00 | 346.35 | 2025-07-16 15:15:00 | 335.60 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-07-23 10:30:00 | 349.80 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-24 10:00:00 | 350.05 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-24 13:00:00 | 349.95 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-24 15:00:00 | 353.85 | 2025-07-25 12:15:00 | 346.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-31 10:00:00 | 331.55 | 2025-07-31 12:15:00 | 338.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-07-31 13:45:00 | 331.70 | 2025-08-01 09:15:00 | 342.75 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-08-06 12:45:00 | 344.40 | 2025-08-07 13:15:00 | 340.30 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-06 15:00:00 | 345.75 | 2025-08-07 13:15:00 | 340.30 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-05 09:15:00 | 333.50 | 2025-09-11 12:15:00 | 339.30 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2025-09-05 11:45:00 | 331.95 | 2025-09-11 12:15:00 | 339.30 | STOP_HIT | 1.00 | 2.21% |
| SELL | retest2 | 2025-09-17 11:15:00 | 330.75 | 2025-09-25 14:15:00 | 315.50 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-09-19 10:30:00 | 332.10 | 2025-09-26 09:15:00 | 314.21 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2025-09-19 15:15:00 | 329.95 | 2025-09-26 09:15:00 | 313.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:15:00 | 330.75 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2025-09-19 10:30:00 | 332.10 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2025-09-19 15:15:00 | 329.95 | 2025-09-29 09:15:00 | 313.55 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2025-10-15 13:00:00 | 310.85 | 2025-10-20 12:15:00 | 312.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-17 11:45:00 | 310.70 | 2025-10-20 12:15:00 | 312.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-14 09:15:00 | 286.30 | 2025-11-14 10:15:00 | 288.80 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-25 09:15:00 | 283.70 | 2025-11-26 11:15:00 | 285.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 10:15:00 | 276.90 | 2025-12-01 14:15:00 | 283.60 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-12-05 14:00:00 | 281.95 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-05 14:45:00 | 281.40 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-08 09:15:00 | 282.85 | 2025-12-08 09:15:00 | 278.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-08 11:45:00 | 281.20 | 2025-12-08 12:15:00 | 279.45 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-22 15:15:00 | 281.80 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-23 11:30:00 | 282.00 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-23 12:00:00 | 282.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-23 13:30:00 | 282.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-26 15:00:00 | 280.75 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-29 09:15:00 | 277.25 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-12-29 15:00:00 | 278.05 | 2025-12-30 14:15:00 | 286.95 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-01-07 13:45:00 | 287.35 | 2026-01-08 11:15:00 | 282.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2026-01-13 11:00:00 | 274.15 | 2026-01-21 09:15:00 | 260.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 11:30:00 | 273.85 | 2026-01-21 09:15:00 | 260.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 12:00:00 | 273.55 | 2026-01-21 09:15:00 | 259.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 14:00:00 | 273.60 | 2026-01-21 09:15:00 | 259.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 11:00:00 | 274.15 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest1 | 2026-01-13 11:30:00 | 273.85 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest1 | 2026-01-13 12:00:00 | 273.55 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest1 | 2026-01-13 14:00:00 | 273.60 | 2026-01-21 15:15:00 | 261.40 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2026-01-20 13:30:00 | 265.70 | 2026-01-23 09:15:00 | 266.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-30 11:00:00 | 273.95 | 2026-02-01 15:15:00 | 266.90 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-30 14:00:00 | 274.35 | 2026-02-01 15:15:00 | 266.90 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-11 09:30:00 | 258.80 | 2026-02-13 09:15:00 | 245.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 258.80 | 2026-02-13 15:15:00 | 250.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest1 | 2026-03-06 10:30:00 | 227.50 | 2026-03-10 09:15:00 | 226.95 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-03-09 09:15:00 | 219.97 | 2026-03-10 14:15:00 | 231.62 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-03-17 12:00:00 | 221.57 | 2026-03-19 13:15:00 | 210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:00:00 | 222.02 | 2026-03-19 13:15:00 | 210.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 10:45:00 | 222.10 | 2026-03-19 13:15:00 | 210.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 12:00:00 | 221.57 | 2026-03-23 09:15:00 | 199.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 14:00:00 | 222.02 | 2026-03-23 09:15:00 | 199.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 10:45:00 | 222.10 | 2026-03-23 09:15:00 | 199.89 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 09:15:00 | 251.55 | 2026-04-29 09:15:00 | 256.35 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2026-05-04 09:30:00 | 265.50 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-05-04 10:45:00 | 265.50 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-05-04 11:45:00 | 265.45 | 2026-05-05 09:15:00 | 257.65 | STOP_HIT | 1.00 | -2.94% |
