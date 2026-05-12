# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 568.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 99 |
| ALERT2 | 97 |
| ALERT2_SKIP | 42 |
| ALERT3 | 251 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 102 |
| PARTIAL | 15 |
| TARGET_HIT | 9 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 49 / 77
- **Target hits / Stop hits / Partials:** 9 / 102 / 15
- **Avg / median % per leg:** 1.06% / -0.64%
- **Sum % (uncompounded):** 133.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 16 | 31.4% | 8 | 43 | 0 | 0.81% | 41.2% |
| BUY @ 2nd Alert (retest1) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.61% | -14.5% |
| BUY @ 3rd Alert (retest2) | 42 | 16 | 38.1% | 8 | 34 | 0 | 1.33% | 55.7% |
| SELL (all) | 75 | 33 | 44.0% | 1 | 59 | 15 | 1.23% | 92.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 75 | 33 | 44.0% | 1 | 59 | 15 | 1.23% | 92.5% |
| retest1 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.61% | -14.5% |
| retest2 (combined) | 117 | 49 | 41.9% | 9 | 93 | 15 | 1.27% | 148.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 376.10 | 367.99 | 367.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 378.30 | 371.45 | 368.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 380.10 | 380.25 | 376.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:45:00 | 379.95 | 380.25 | 376.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 374.10 | 378.64 | 377.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 375.05 | 378.64 | 377.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 378.75 | 378.66 | 377.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 380.00 | 378.63 | 377.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 375.50 | 386.21 | 387.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 375.50 | 386.21 | 387.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 365.50 | 373.40 | 377.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 373.60 | 371.80 | 375.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:00:00 | 373.60 | 371.80 | 375.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 369.00 | 371.24 | 374.70 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 377.40 | 375.10 | 374.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 379.30 | 375.94 | 375.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 376.25 | 376.64 | 375.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 376.25 | 376.64 | 375.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 366.50 | 374.59 | 374.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 363.70 | 371.13 | 373.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 360.90 | 359.91 | 364.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:30:00 | 361.25 | 359.91 | 364.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 365.35 | 359.76 | 363.21 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 372.35 | 366.12 | 365.58 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 350.80 | 365.32 | 365.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 314.95 | 355.24 | 361.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 316.80 | 313.81 | 329.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 316.80 | 313.81 | 329.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 338.35 | 322.02 | 328.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 341.65 | 322.02 | 328.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 343.05 | 326.22 | 329.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 343.05 | 326.22 | 329.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 339.55 | 332.22 | 332.08 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 330.20 | 336.15 | 336.50 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 338.80 | 336.19 | 335.88 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 333.15 | 335.73 | 336.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 331.85 | 333.55 | 334.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 335.50 | 330.27 | 331.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 335.50 | 330.27 | 331.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 335.50 | 330.27 | 331.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 335.50 | 330.27 | 331.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 336.60 | 331.54 | 332.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 336.60 | 331.54 | 332.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 333.75 | 332.70 | 332.67 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 331.00 | 332.69 | 332.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 325.10 | 330.90 | 331.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 13:15:00 | 330.85 | 330.18 | 331.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 13:15:00 | 330.85 | 330.18 | 331.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 330.85 | 330.18 | 331.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 330.85 | 330.18 | 331.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 331.65 | 330.47 | 331.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 331.65 | 330.47 | 331.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 332.70 | 330.92 | 331.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 338.90 | 330.92 | 331.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 330.60 | 331.24 | 331.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 329.00 | 330.65 | 331.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 326.50 | 322.61 | 322.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 326.50 | 322.61 | 322.40 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 319.75 | 322.04 | 322.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 317.95 | 321.22 | 321.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 324.00 | 320.44 | 321.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 324.00 | 320.44 | 321.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 324.00 | 320.44 | 321.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 325.45 | 320.44 | 321.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 324.40 | 321.69 | 321.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 327.25 | 325.22 | 323.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 325.00 | 325.93 | 324.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 325.00 | 325.93 | 324.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 325.00 | 325.93 | 324.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 322.95 | 325.93 | 324.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 329.20 | 326.58 | 325.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:45:00 | 330.30 | 327.38 | 325.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:45:00 | 330.45 | 328.04 | 326.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 330.90 | 328.61 | 326.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:00:00 | 331.20 | 329.13 | 326.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 332.60 | 342.83 | 339.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 332.60 | 342.83 | 339.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 332.45 | 340.76 | 338.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 327.80 | 340.76 | 338.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-10 13:15:00 | 332.00 | 336.78 | 337.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 332.00 | 336.78 | 337.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 330.35 | 332.53 | 334.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 335.35 | 332.78 | 333.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 335.35 | 332.78 | 333.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 335.35 | 332.78 | 333.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 335.35 | 332.78 | 333.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 333.25 | 332.87 | 333.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 332.70 | 332.87 | 333.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 332.80 | 332.96 | 333.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 13:00:00 | 332.40 | 332.84 | 333.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 15:00:00 | 331.05 | 332.57 | 333.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 332.75 | 332.53 | 333.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:45:00 | 333.30 | 332.53 | 333.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 333.60 | 332.75 | 333.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 333.25 | 332.75 | 333.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 330.90 | 332.38 | 333.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:30:00 | 330.30 | 331.99 | 332.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 316.06 | 320.89 | 325.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 316.16 | 320.89 | 325.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 315.78 | 320.89 | 325.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 314.50 | 320.89 | 325.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 313.78 | 320.89 | 325.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 314.05 | 312.71 | 318.14 | SL hit (close>ema200) qty=0.50 sl=312.71 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 316.90 | 312.61 | 312.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 319.80 | 315.54 | 314.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 316.10 | 316.69 | 315.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 316.10 | 316.69 | 315.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 316.25 | 316.60 | 315.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:15:00 | 316.20 | 316.60 | 315.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 316.20 | 316.52 | 315.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 318.25 | 316.52 | 315.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 319.70 | 317.16 | 315.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 323.55 | 317.88 | 316.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:00:00 | 321.80 | 319.35 | 317.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:00:00 | 321.80 | 320.41 | 318.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 311.90 | 319.06 | 319.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 311.90 | 319.06 | 319.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 300.95 | 310.43 | 314.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 289.70 | 289.24 | 295.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:00:00 | 289.70 | 289.24 | 295.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 295.70 | 291.83 | 294.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 295.70 | 291.83 | 294.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 297.00 | 292.86 | 294.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 293.10 | 292.86 | 294.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 299.25 | 295.00 | 295.32 | SL hit (close>static) qty=1.00 sl=297.50 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 13:15:00 | 298.45 | 295.69 | 295.61 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 294.90 | 295.51 | 295.54 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 302.90 | 296.99 | 296.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 312.20 | 304.21 | 301.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 309.95 | 310.14 | 306.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 309.95 | 310.14 | 306.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 304.40 | 309.92 | 306.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 303.15 | 309.92 | 306.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 300.95 | 308.13 | 306.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 300.95 | 308.13 | 306.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 298.25 | 304.23 | 304.83 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 307.75 | 304.93 | 304.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 314.50 | 306.92 | 305.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 318.40 | 319.58 | 315.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:30:00 | 319.10 | 319.58 | 315.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 318.10 | 318.81 | 316.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 317.75 | 318.81 | 316.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 316.00 | 318.25 | 316.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:30:00 | 316.80 | 318.25 | 316.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 315.95 | 317.79 | 316.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:15:00 | 317.00 | 317.79 | 316.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 318.30 | 317.46 | 316.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:30:00 | 316.80 | 317.56 | 317.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 319.00 | 317.84 | 317.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 318.00 | 317.88 | 317.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 317.10 | 317.88 | 317.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 317.25 | 317.75 | 317.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 315.45 | 316.93 | 317.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 315.45 | 316.93 | 317.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 313.90 | 316.32 | 316.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 316.20 | 315.59 | 316.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 316.20 | 315.59 | 316.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 316.20 | 315.59 | 316.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 322.80 | 315.59 | 316.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 323.50 | 317.17 | 316.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 330.00 | 323.84 | 320.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 15:15:00 | 329.45 | 329.85 | 326.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:15:00 | 330.25 | 329.85 | 326.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 332.00 | 330.28 | 326.85 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 322.00 | 326.88 | 327.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 321.50 | 325.26 | 326.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 330.00 | 326.21 | 326.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 330.00 | 326.21 | 326.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 330.00 | 326.21 | 326.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 330.00 | 326.21 | 326.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 328.00 | 326.57 | 326.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 326.85 | 326.57 | 326.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 330.25 | 327.47 | 327.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 331.40 | 328.78 | 327.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 330.35 | 330.49 | 328.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 09:15:00 | 325.20 | 330.49 | 328.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 325.35 | 329.46 | 328.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 326.80 | 329.46 | 328.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 324.75 | 328.52 | 328.30 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 323.75 | 327.57 | 327.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 321.75 | 325.59 | 326.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 14:15:00 | 316.70 | 313.19 | 315.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 316.70 | 313.19 | 315.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 316.70 | 313.19 | 315.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 316.70 | 313.19 | 315.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 314.50 | 313.45 | 315.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 313.25 | 312.50 | 315.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:45:00 | 313.45 | 308.64 | 309.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:45:00 | 312.75 | 309.39 | 310.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 315.50 | 311.14 | 310.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 315.50 | 311.14 | 310.81 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 309.50 | 311.03 | 311.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 308.35 | 310.49 | 310.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 310.25 | 309.48 | 310.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 10:15:00 | 310.25 | 309.48 | 310.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 310.25 | 309.48 | 310.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 310.25 | 309.48 | 310.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 312.00 | 309.99 | 310.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 312.00 | 309.99 | 310.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 311.90 | 310.37 | 310.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:45:00 | 312.60 | 310.37 | 310.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 312.15 | 310.72 | 310.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 315.15 | 312.53 | 311.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 314.50 | 314.61 | 313.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:15:00 | 316.60 | 314.61 | 313.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:00:00 | 316.35 | 314.96 | 313.52 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:00:00 | 315.75 | 315.12 | 313.72 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 313.35 | 314.76 | 313.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 313.35 | 314.76 | 313.69 | SL hit (close<ema400) qty=1.00 sl=313.69 alert=retest1 |

### Cycle 32 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 314.55 | 317.53 | 317.78 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 323.20 | 317.80 | 317.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 325.95 | 319.43 | 318.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 15:15:00 | 339.00 | 339.78 | 336.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:15:00 | 337.55 | 339.78 | 336.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 336.20 | 339.07 | 336.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 333.95 | 339.07 | 336.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 337.15 | 338.68 | 336.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 342.50 | 339.12 | 336.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 337.60 | 342.80 | 343.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 337.60 | 342.80 | 343.03 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 349.10 | 343.75 | 343.34 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 338.00 | 342.80 | 343.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 333.30 | 339.35 | 341.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 316.70 | 316.48 | 322.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 316.70 | 316.48 | 322.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 316.00 | 315.45 | 317.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 313.10 | 314.98 | 317.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 313.15 | 314.84 | 317.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 312.50 | 314.94 | 315.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 15:15:00 | 324.20 | 316.08 | 315.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 324.20 | 316.08 | 315.70 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 317.10 | 318.64 | 318.65 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 319.30 | 318.77 | 318.71 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 314.00 | 317.82 | 318.28 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 322.50 | 318.92 | 318.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 324.25 | 319.98 | 319.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 318.35 | 319.82 | 319.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 318.35 | 319.82 | 319.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 318.35 | 319.82 | 319.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 317.00 | 319.82 | 319.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 317.60 | 319.38 | 319.00 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 316.05 | 318.71 | 318.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 315.45 | 318.06 | 318.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 278.90 | 277.57 | 285.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:45:00 | 279.25 | 277.57 | 285.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 285.85 | 279.23 | 285.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 285.85 | 279.23 | 285.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 288.45 | 281.07 | 285.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 286.95 | 281.07 | 285.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 290.00 | 282.86 | 285.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 290.00 | 282.86 | 285.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 288.35 | 286.50 | 286.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 288.35 | 286.50 | 286.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 283.00 | 285.80 | 286.51 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 289.55 | 287.16 | 286.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 292.50 | 288.23 | 287.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 291.00 | 291.55 | 290.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 12:15:00 | 291.00 | 291.55 | 290.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 291.00 | 291.55 | 290.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 290.70 | 291.55 | 290.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 294.55 | 291.98 | 290.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 292.25 | 291.98 | 290.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 288.50 | 292.38 | 291.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 288.50 | 292.38 | 291.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 287.15 | 291.33 | 290.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 287.15 | 291.33 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 289.90 | 291.05 | 290.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 294.45 | 291.73 | 291.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 291.65 | 291.71 | 291.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 292.30 | 291.83 | 291.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 293.50 | 291.46 | 291.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 289.40 | 291.05 | 291.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 289.40 | 291.05 | 291.05 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 292.20 | 291.26 | 291.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 299.00 | 292.81 | 291.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 10:15:00 | 294.60 | 295.30 | 293.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 294.60 | 295.30 | 293.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 294.60 | 295.30 | 293.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 293.25 | 295.30 | 293.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 294.15 | 296.84 | 295.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 295.65 | 296.84 | 295.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 292.15 | 295.90 | 295.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 290.50 | 295.90 | 295.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 291.80 | 294.39 | 294.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 289.75 | 292.40 | 293.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 264.50 | 264.12 | 268.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 264.50 | 264.12 | 268.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 265.65 | 263.81 | 265.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 264.85 | 263.81 | 265.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 266.40 | 264.33 | 266.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 266.40 | 264.33 | 266.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 264.90 | 264.44 | 265.90 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 268.65 | 266.60 | 266.53 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 263.40 | 266.47 | 266.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 260.35 | 264.60 | 265.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 263.50 | 263.08 | 264.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 263.50 | 263.08 | 264.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 263.30 | 263.12 | 264.12 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 271.15 | 265.25 | 264.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 272.95 | 266.79 | 265.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 274.40 | 277.27 | 274.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 274.40 | 277.27 | 274.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 276.60 | 277.14 | 274.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:30:00 | 277.15 | 277.14 | 274.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 275.80 | 276.87 | 274.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 275.30 | 276.87 | 274.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 274.85 | 277.05 | 276.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 274.85 | 277.05 | 276.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 275.10 | 276.66 | 275.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 275.50 | 276.66 | 275.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 274.70 | 276.27 | 275.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 274.70 | 276.27 | 275.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 273.10 | 275.42 | 275.50 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 275.60 | 275.28 | 275.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 276.65 | 275.72 | 275.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 282.75 | 282.89 | 280.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:00:00 | 282.75 | 282.89 | 280.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 284.50 | 283.07 | 281.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 281.40 | 283.07 | 281.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 284.20 | 283.78 | 282.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 282.60 | 283.78 | 282.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 289.90 | 290.69 | 289.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 289.25 | 290.69 | 289.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 292.65 | 291.08 | 289.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 293.05 | 291.23 | 289.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:30:00 | 292.90 | 291.91 | 290.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 15:15:00 | 289.05 | 290.87 | 290.40 | SL hit (close<static) qty=1.00 sl=289.25 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 287.20 | 289.98 | 290.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 286.00 | 289.18 | 289.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 290.80 | 288.88 | 289.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 14:15:00 | 290.80 | 288.88 | 289.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 290.80 | 288.88 | 289.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 290.80 | 288.88 | 289.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 289.55 | 289.02 | 289.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 285.15 | 289.02 | 289.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:30:00 | 288.70 | 288.00 | 288.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 288.70 | 288.08 | 288.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 289.60 | 288.42 | 288.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 289.60 | 288.42 | 288.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 291.50 | 289.21 | 288.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 15:15:00 | 288.75 | 289.12 | 288.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 09:15:00 | 288.00 | 289.12 | 288.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 287.15 | 288.72 | 288.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 287.25 | 288.72 | 288.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 287.25 | 288.43 | 288.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 284.95 | 287.34 | 287.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 281.35 | 280.13 | 283.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 14:15:00 | 281.35 | 280.13 | 283.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 281.35 | 280.13 | 283.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 281.35 | 280.13 | 283.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 279.55 | 277.30 | 279.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 279.55 | 277.30 | 279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 278.00 | 277.44 | 279.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 275.85 | 277.44 | 279.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 276.15 | 277.11 | 279.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 280.55 | 277.80 | 279.38 | SL hit (close>static) qty=1.00 sl=280.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 284.20 | 276.87 | 276.26 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 265.95 | 276.68 | 276.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 264.50 | 272.57 | 274.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 249.80 | 249.01 | 253.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 249.80 | 249.01 | 253.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 248.78 | 248.11 | 249.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:30:00 | 248.35 | 248.11 | 249.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 250.50 | 248.59 | 249.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:00:00 | 250.50 | 248.59 | 249.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 249.70 | 248.81 | 249.75 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 255.90 | 250.86 | 250.45 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 245.75 | 250.01 | 250.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 240.18 | 248.04 | 249.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 240.95 | 240.45 | 243.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 240.95 | 240.45 | 243.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 239.18 | 237.86 | 239.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 239.18 | 237.86 | 239.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 239.40 | 236.79 | 238.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:00:00 | 239.40 | 236.79 | 238.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 238.20 | 237.07 | 238.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:30:00 | 238.93 | 237.07 | 238.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 236.90 | 237.04 | 237.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 235.99 | 237.04 | 237.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 224.19 | 229.21 | 233.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 226.41 | 224.59 | 228.13 | SL hit (close>ema200) qty=0.50 sl=224.59 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 231.57 | 227.77 | 227.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 237.21 | 230.13 | 228.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 246.72 | 247.56 | 242.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 246.72 | 247.56 | 242.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 245.23 | 247.09 | 242.81 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 238.70 | 242.24 | 242.37 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 245.30 | 242.51 | 242.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 245.96 | 243.46 | 242.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 243.50 | 244.44 | 243.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 243.50 | 244.44 | 243.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 243.50 | 244.44 | 243.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 243.50 | 244.44 | 243.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 244.94 | 244.54 | 243.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 245.63 | 244.54 | 243.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 242.40 | 244.47 | 244.00 | SL hit (close<static) qty=1.00 sl=242.45 alert=retest2 |

### Cycle 62 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 233.99 | 241.96 | 242.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 232.98 | 240.16 | 242.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 231.70 | 231.36 | 235.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 231.70 | 231.36 | 235.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 233.90 | 230.79 | 233.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 233.90 | 230.79 | 233.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 230.64 | 230.76 | 233.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 229.68 | 230.76 | 233.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 234.79 | 229.94 | 231.53 | SL hit (close>static) qty=1.00 sl=234.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 238.30 | 232.83 | 232.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 239.01 | 234.07 | 232.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 237.42 | 237.52 | 235.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 237.42 | 237.52 | 235.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 237.42 | 237.52 | 235.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 236.22 | 237.52 | 235.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 235.42 | 237.10 | 235.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 236.21 | 237.10 | 235.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 241.24 | 237.93 | 235.92 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 231.61 | 234.93 | 235.29 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 238.75 | 235.80 | 235.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 245.00 | 239.18 | 237.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 242.45 | 243.74 | 241.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 242.45 | 243.74 | 241.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 241.94 | 243.38 | 241.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 241.94 | 243.38 | 241.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 239.73 | 242.65 | 241.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 239.70 | 242.65 | 241.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 240.69 | 242.26 | 241.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 242.25 | 241.94 | 241.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 239.32 | 242.36 | 241.98 | SL hit (close<static) qty=1.00 sl=239.41 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 237.46 | 240.98 | 241.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 236.83 | 239.62 | 240.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 230.71 | 227.18 | 230.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 230.71 | 227.18 | 230.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 230.71 | 227.18 | 230.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 230.71 | 227.18 | 230.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 230.98 | 227.94 | 230.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 230.98 | 227.94 | 230.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 230.16 | 228.38 | 230.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 230.16 | 228.38 | 230.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 229.99 | 228.70 | 230.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 229.35 | 228.70 | 230.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:15:00 | 217.88 | 223.72 | 227.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 215.85 | 214.71 | 218.63 | SL hit (close>ema200) qty=0.50 sl=214.71 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 219.53 | 216.60 | 216.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 222.24 | 218.74 | 217.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 221.69 | 222.94 | 221.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 221.69 | 222.94 | 221.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 221.69 | 222.94 | 221.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 218.45 | 222.94 | 221.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 219.17 | 222.19 | 221.29 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 218.95 | 220.78 | 220.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 216.51 | 218.94 | 219.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 203.00 | 201.82 | 206.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 203.00 | 201.82 | 206.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 204.21 | 202.92 | 205.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 205.21 | 202.92 | 205.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 205.02 | 203.34 | 205.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 205.02 | 203.34 | 205.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 209.85 | 204.00 | 204.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 209.85 | 204.00 | 204.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 212.03 | 205.61 | 205.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 213.14 | 208.10 | 206.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 224.88 | 225.19 | 220.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 13:30:00 | 226.25 | 225.21 | 220.72 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 14:30:00 | 226.60 | 224.97 | 221.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 221.74 | 223.93 | 221.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 223.10 | 223.93 | 221.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 222.80 | 223.70 | 221.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:45:00 | 222.84 | 223.70 | 221.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 220.76 | 223.11 | 221.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 220.76 | 223.11 | 221.30 | SL hit (close<ema400) qty=1.00 sl=221.30 alert=retest1 |

### Cycle 70 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 216.90 | 220.15 | 220.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 214.50 | 219.02 | 219.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 217.92 | 216.77 | 218.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 217.92 | 216.77 | 218.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 217.92 | 216.77 | 218.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 217.27 | 216.77 | 218.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 219.00 | 217.21 | 218.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 219.16 | 217.21 | 218.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 220.90 | 217.95 | 218.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 222.15 | 217.95 | 218.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 215.83 | 217.53 | 218.20 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 221.60 | 218.79 | 218.55 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 215.38 | 217.97 | 218.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 11:15:00 | 214.50 | 216.47 | 217.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 218.99 | 216.34 | 216.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 218.99 | 216.34 | 216.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 218.99 | 216.34 | 216.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 218.99 | 216.34 | 216.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 219.47 | 217.28 | 217.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 221.30 | 218.65 | 217.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 222.50 | 222.60 | 220.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 14:00:00 | 222.50 | 222.60 | 220.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 231.82 | 234.40 | 232.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 231.82 | 234.40 | 232.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 230.35 | 233.59 | 231.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 230.48 | 233.59 | 231.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 229.40 | 232.75 | 231.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 228.30 | 232.75 | 231.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 228.96 | 231.28 | 231.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 228.96 | 231.28 | 231.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 228.00 | 230.62 | 230.90 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 235.30 | 231.56 | 231.30 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 228.40 | 230.86 | 231.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 225.88 | 229.50 | 230.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 224.60 | 224.43 | 226.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 224.60 | 224.43 | 226.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 228.55 | 225.31 | 226.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 228.55 | 225.31 | 226.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 223.85 | 225.02 | 226.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 223.60 | 225.02 | 226.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:00:00 | 223.02 | 224.62 | 226.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 226.35 | 223.42 | 223.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 226.35 | 223.42 | 223.06 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 210.94 | 222.31 | 223.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 208.45 | 217.80 | 220.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 192.85 | 192.22 | 199.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 192.85 | 192.22 | 199.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 196.21 | 192.21 | 194.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 196.21 | 192.21 | 194.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 195.80 | 192.93 | 194.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 200.19 | 192.93 | 194.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 201.90 | 196.87 | 196.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 206.09 | 200.00 | 198.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 209.40 | 209.55 | 206.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 209.24 | 209.55 | 206.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 220.98 | 222.68 | 219.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 219.47 | 222.68 | 219.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 218.26 | 223.02 | 222.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 218.26 | 223.02 | 222.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 215.64 | 221.54 | 221.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 213.21 | 217.40 | 219.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 218.42 | 216.96 | 218.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 218.42 | 216.96 | 218.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 218.38 | 217.25 | 218.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 217.75 | 218.09 | 218.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 217.20 | 218.21 | 218.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:30:00 | 217.56 | 217.96 | 218.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 217.21 | 217.96 | 218.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 218.82 | 218.13 | 218.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 218.82 | 218.13 | 218.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 218.06 | 218.12 | 218.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 216.57 | 218.12 | 218.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 217.48 | 218.00 | 218.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:30:00 | 217.27 | 217.66 | 218.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 217.30 | 216.12 | 217.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 216.21 | 216.14 | 217.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 215.17 | 216.14 | 217.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 217.60 | 216.07 | 216.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 217.60 | 216.07 | 216.00 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 214.04 | 215.66 | 215.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 211.68 | 214.52 | 215.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 209.61 | 209.55 | 211.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 210.20 | 209.55 | 211.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 211.32 | 209.91 | 211.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 211.32 | 209.91 | 211.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 212.23 | 210.38 | 211.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 212.23 | 210.38 | 211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 211.67 | 210.63 | 211.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 212.01 | 210.63 | 211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 213.34 | 211.18 | 211.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 210.92 | 211.51 | 211.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 200.37 | 207.00 | 209.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 206.00 | 205.21 | 207.19 | SL hit (close>ema200) qty=0.50 sl=205.21 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 217.74 | 209.67 | 208.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 220.89 | 213.13 | 210.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 221.68 | 222.05 | 218.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 221.68 | 222.05 | 218.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 227.75 | 229.09 | 226.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 230.85 | 229.09 | 226.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 229.16 | 229.10 | 226.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 226.11 | 227.89 | 227.15 | SL hit (close<static) qty=1.00 sl=226.61 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 224.90 | 227.96 | 227.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 224.49 | 227.04 | 227.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 227.05 | 227.04 | 227.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 227.05 | 227.04 | 227.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 227.84 | 227.20 | 227.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 227.84 | 227.20 | 227.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 229.30 | 227.62 | 227.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 229.30 | 227.62 | 227.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 228.90 | 227.88 | 227.77 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 226.51 | 227.60 | 227.65 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 228.40 | 227.71 | 227.69 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 226.01 | 227.37 | 227.54 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 229.70 | 227.53 | 227.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 234.71 | 228.97 | 228.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 244.55 | 244.89 | 241.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:15:00 | 256.23 | 244.89 | 241.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 248.40 | 250.18 | 248.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 248.40 | 250.18 | 248.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 248.78 | 249.90 | 248.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 247.72 | 249.90 | 248.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 247.95 | 249.51 | 248.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 247.95 | 249.51 | 248.24 | SL hit (close<ema400) qty=1.00 sl=248.24 alert=retest1 |

### Cycle 90 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 246.73 | 247.84 | 247.86 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 250.62 | 248.39 | 248.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 251.99 | 249.11 | 248.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 249.40 | 250.69 | 249.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 249.35 | 250.42 | 249.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:30:00 | 249.00 | 250.42 | 249.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 248.99 | 249.76 | 249.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 249.67 | 249.76 | 249.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 254.80 | 259.79 | 260.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 254.80 | 259.79 | 260.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 11:15:00 | 254.50 | 256.48 | 258.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 256.64 | 254.17 | 255.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 261.51 | 255.63 | 256.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 261.51 | 255.63 | 256.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 263.00 | 257.11 | 256.99 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 253.05 | 257.00 | 257.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 249.46 | 253.26 | 254.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 249.03 | 247.49 | 250.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 249.03 | 247.49 | 250.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 247.99 | 247.76 | 249.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 248.40 | 248.08 | 249.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 249.89 | 248.44 | 249.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 249.89 | 248.44 | 249.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 250.45 | 248.84 | 249.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 250.45 | 248.84 | 249.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 252.80 | 249.63 | 249.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 252.81 | 249.63 | 249.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 252.18 | 250.14 | 249.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 258.27 | 252.14 | 250.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 276.25 | 278.47 | 274.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 276.85 | 278.47 | 274.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 281.90 | 282.80 | 280.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 281.60 | 282.80 | 280.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 281.00 | 282.14 | 281.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 281.40 | 282.14 | 281.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 280.00 | 281.71 | 280.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 280.95 | 281.71 | 280.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 280.80 | 281.53 | 280.96 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 277.45 | 280.33 | 280.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 275.95 | 279.45 | 280.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 275.00 | 274.10 | 275.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 13:00:00 | 275.00 | 274.10 | 275.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 264.35 | 272.01 | 274.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 263.65 | 269.54 | 272.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 263.55 | 267.52 | 271.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 263.55 | 265.78 | 269.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:45:00 | 263.20 | 265.31 | 268.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 265.40 | 265.10 | 267.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 267.40 | 265.10 | 267.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 264.30 | 263.46 | 264.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 264.50 | 263.46 | 264.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 264.50 | 263.67 | 264.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:15:00 | 264.60 | 263.67 | 264.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 263.50 | 263.63 | 264.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 264.45 | 263.63 | 264.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 261.90 | 263.19 | 264.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:30:00 | 261.05 | 262.42 | 263.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 260.20 | 262.56 | 262.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 265.35 | 263.28 | 263.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 265.35 | 263.28 | 263.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 13:15:00 | 267.10 | 264.83 | 264.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 273.85 | 274.68 | 272.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 273.85 | 274.68 | 272.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 272.50 | 274.24 | 272.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 272.50 | 274.24 | 272.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 271.55 | 273.70 | 272.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 278.00 | 273.70 | 272.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 266.95 | 272.85 | 273.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 266.95 | 272.85 | 273.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 264.90 | 271.26 | 272.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 254.80 | 253.57 | 258.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 254.80 | 253.57 | 258.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 259.10 | 255.38 | 258.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 259.10 | 255.38 | 258.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 258.50 | 256.00 | 258.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 259.65 | 256.00 | 258.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 258.40 | 256.48 | 258.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 258.40 | 256.48 | 258.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 258.20 | 256.83 | 258.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 258.05 | 256.83 | 258.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 256.65 | 256.79 | 258.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 254.20 | 256.84 | 258.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 261.90 | 258.08 | 258.48 | SL hit (close>static) qty=1.00 sl=260.80 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 245.49 | 243.95 | 243.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 14:15:00 | 245.64 | 244.32 | 244.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 244.48 | 244.25 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 241.96 | 243.79 | 243.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 240.46 | 243.12 | 243.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 12:15:00 | 237.60 | 237.50 | 239.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 13:00:00 | 237.60 | 237.50 | 239.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 237.35 | 237.52 | 238.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 238.40 | 237.52 | 238.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 237.80 | 237.62 | 238.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 237.80 | 237.62 | 238.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 236.60 | 237.42 | 238.45 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 242.66 | 239.39 | 239.06 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 240.00 | 240.90 | 240.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 239.11 | 240.43 | 240.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 238.77 | 238.24 | 239.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 238.91 | 238.38 | 239.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 237.17 | 238.45 | 239.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 232.50 | 231.48 | 231.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 232.50 | 231.48 | 231.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 234.65 | 232.12 | 231.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 245.76 | 245.92 | 242.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 246.45 | 245.92 | 242.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 242.74 | 244.89 | 242.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 242.74 | 244.89 | 242.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 243.35 | 244.58 | 242.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 243.45 | 244.58 | 242.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 241.59 | 243.98 | 242.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 241.59 | 243.98 | 242.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 242.29 | 243.65 | 242.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 244.22 | 242.95 | 242.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 245.14 | 244.70 | 243.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 243.60 | 243.86 | 243.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 11:15:00 | 267.96 | 256.55 | 251.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 279.60 | 281.37 | 281.60 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 283.99 | 281.98 | 281.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 301.56 | 285.89 | 283.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 308.10 | 308.46 | 303.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:15:00 | 316.18 | 308.46 | 303.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 316.76 | 320.80 | 315.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 311.53 | 318.14 | 315.36 | SL hit (close<ema400) qty=1.00 sl=315.36 alert=retest1 |

### Cycle 106 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 332.00 | 335.83 | 336.04 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 339.15 | 336.49 | 336.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 10:15:00 | 339.40 | 337.07 | 336.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 344.95 | 355.09 | 349.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 344.50 | 352.97 | 349.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 343.20 | 352.97 | 349.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 342.70 | 346.64 | 347.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 337.00 | 340.97 | 343.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 335.20 | 334.02 | 338.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 335.20 | 334.02 | 338.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 341.10 | 335.44 | 338.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 341.10 | 335.44 | 338.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 341.40 | 336.63 | 338.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 341.30 | 336.63 | 338.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 341.50 | 339.57 | 339.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 342.45 | 340.47 | 340.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 346.00 | 342.00 | 341.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 344.30 | 342.46 | 341.36 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 337.20 | 340.55 | 340.94 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 344.50 | 341.42 | 341.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 346.90 | 342.52 | 341.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 334.40 | 341.28 | 341.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 332.00 | 339.43 | 340.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 330.60 | 335.39 | 338.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 343.25 | 335.91 | 337.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 339.60 | 336.65 | 337.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 340.65 | 336.65 | 337.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 342.40 | 338.69 | 338.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 343.75 | 340.42 | 339.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 346.30 | 346.44 | 344.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 346.30 | 346.44 | 344.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 344.25 | 345.97 | 344.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 344.30 | 345.97 | 344.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 345.50 | 345.88 | 344.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 352.15 | 344.29 | 343.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 347.25 | 348.73 | 347.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 341.85 | 346.58 | 347.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 341.85 | 346.58 | 347.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 340.70 | 343.62 | 345.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 340.05 | 342.30 | 343.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:15:00 | 323.05 | 331.33 | 336.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 326.35 | 325.64 | 330.60 | SL hit (close>ema200) qty=0.50 sl=325.64 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 341.50 | 333.48 | 332.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 344.00 | 338.09 | 335.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 343.70 | 348.53 | 343.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 342.65 | 347.36 | 343.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 342.65 | 347.36 | 343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 343.25 | 346.54 | 343.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 345.75 | 343.45 | 342.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 340.15 | 342.32 | 342.30 | SL hit (close<static) qty=1.00 sl=340.85 alert=retest2 |

### Cycle 116 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 337.65 | 341.39 | 341.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 334.40 | 339.41 | 340.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 337.55 | 335.92 | 337.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 339.85 | 336.71 | 337.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 339.85 | 336.71 | 337.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 339.10 | 337.19 | 337.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 337.65 | 337.49 | 338.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 337.45 | 337.49 | 338.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 320.77 | 325.28 | 328.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 320.58 | 325.28 | 328.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 322.00 | 314.60 | 317.85 | SL hit (close>ema200) qty=0.50 sl=314.60 alert=retest2 |

### Cycle 117 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 321.45 | 319.21 | 319.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 325.05 | 320.49 | 319.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 326.80 | 327.06 | 324.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 326.80 | 327.06 | 324.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 324.90 | 326.70 | 325.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 324.20 | 326.70 | 325.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 325.40 | 326.44 | 325.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 328.20 | 327.53 | 325.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 336.30 | 327.76 | 326.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-04 10:15:00 | 361.02 | 344.62 | 340.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 358.90 | 362.26 | 362.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 356.45 | 361.26 | 361.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 369.60 | 359.73 | 360.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 376.60 | 363.10 | 361.81 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 369.40 | 372.00 | 372.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 366.00 | 370.11 | 371.08 | Break + close below crossover candle low |

### Cycle 121 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 379.50 | 371.24 | 371.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 10:15:00 | 384.80 | 373.95 | 372.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 381.85 | 382.28 | 378.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:45:00 | 381.95 | 382.28 | 378.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 555.90 | 561.15 | 553.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 554.10 | 561.15 | 553.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 554.00 | 559.18 | 553.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 554.00 | 559.18 | 553.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 556.15 | 558.57 | 553.97 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 532.20 | 550.74 | 551.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 523.50 | 536.10 | 543.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:15:00 | 538.00 | 525.17 | 531.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 537.00 | 527.53 | 531.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 540.20 | 527.53 | 531.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 535.50 | 529.13 | 532.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:15:00 | 539.25 | 529.13 | 532.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 541.20 | 531.54 | 532.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 539.70 | 531.54 | 532.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 546.20 | 535.65 | 534.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 547.20 | 537.96 | 535.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 539.40 | 541.86 | 538.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 539.35 | 541.51 | 539.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 552.25 | 541.51 | 539.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 548.35 | 555.48 | 555.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 548.35 | 555.48 | 555.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 540.95 | 551.49 | 553.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 10:15:00 | 543.25 | 540.55 | 545.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 11:00:00 | 543.25 | 540.55 | 545.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 549.70 | 542.38 | 546.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 549.70 | 542.38 | 546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 551.20 | 544.15 | 546.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 551.20 | 544.15 | 546.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 557.00 | 547.41 | 547.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 555.95 | 547.41 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 558.00 | 549.52 | 548.62 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 09:15:00 | 531.00 | 545.82 | 547.02 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 12:15:00 | 546.25 | 543.33 | 543.10 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 540.85 | 542.84 | 542.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 535.00 | 541.27 | 542.17 | Break + close below crossover candle low |

### Cycle 129 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 551.25 | 542.90 | 542.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 561.60 | 550.68 | 546.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 708.55 | 722.05 | 671.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 687.90 | 701.15 | 679.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 687.90 | 701.15 | 679.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 674.00 | 701.15 | 679.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 675.10 | 695.94 | 678.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 571.60 | 695.94 | 678.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 604.65 | 677.68 | 672.03 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 614.50 | 665.04 | 666.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 593.20 | 650.68 | 660.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 610.85 | 606.37 | 623.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 610.85 | 606.37 | 623.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 614.55 | 608.58 | 621.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 593.00 | 617.74 | 620.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:00:00 | 605.70 | 605.50 | 611.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 591.65 | 606.42 | 611.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 606.35 | 600.05 | 599.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 606.35 | 600.05 | 599.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 617.75 | 603.59 | 600.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 604.60 | 615.34 | 610.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 601.85 | 612.64 | 609.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 601.85 | 612.64 | 609.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 599.00 | 606.06 | 606.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 591.40 | 603.13 | 605.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 567.25 | 559.79 | 571.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 567.25 | 559.79 | 571.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 567.55 | 561.34 | 571.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 565.20 | 562.72 | 570.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 575.40 | 565.26 | 571.27 | SL hit (close>static) qty=1.00 sl=572.50 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 576.90 | 567.05 | 566.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 581.95 | 570.03 | 567.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 580.80 | 582.06 | 577.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 14:45:00 | 583.85 | 582.32 | 577.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 585.15 | 582.46 | 578.14 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 586.15 | 583.20 | 578.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 578.60 | 582.10 | 579.11 | SL hit (close<ema400) qty=1.00 sl=579.11 alert=retest1 |

### Cycle 134 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 570.00 | 576.84 | 577.45 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 584.20 | 578.32 | 578.06 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 574.80 | 577.61 | 577.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 573.60 | 576.81 | 577.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 560.80 | 575.64 | 576.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 532.76 | 541.71 | 546.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 504.72 | 525.18 | 534.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 489.20 | 482.88 | 482.33 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 463.30 | 478.96 | 480.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 457.45 | 474.66 | 478.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 459.95 | 455.80 | 463.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 459.75 | 455.80 | 463.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 467.00 | 458.04 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 467.00 | 458.04 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 464.40 | 459.31 | 463.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 463.65 | 459.31 | 463.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 463.00 | 460.05 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 483.45 | 460.05 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 482.05 | 464.45 | 465.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:15:00 | 484.25 | 464.45 | 465.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 484.40 | 468.44 | 467.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 13:15:00 | 489.40 | 478.37 | 474.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 471.60 | 481.31 | 477.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 465.85 | 478.22 | 476.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 465.85 | 478.22 | 476.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 463.60 | 473.87 | 474.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 458.75 | 470.84 | 473.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 482.35 | 468.26 | 470.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 498.40 | 474.29 | 473.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 510.15 | 485.42 | 478.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 475.05 | 487.83 | 482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 471.25 | 484.52 | 481.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 470.35 | 484.52 | 481.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 502.70 | 491.56 | 486.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:45:00 | 508.25 | 501.56 | 494.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 508.30 | 503.51 | 496.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 508.40 | 504.72 | 498.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 522.95 | 504.61 | 499.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 519.95 | 527.25 | 524.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 523.35 | 527.25 | 524.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 559.08 | 534.25 | 529.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 554.25 | 558.24 | 558.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 552.90 | 556.70 | 557.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 557.25 | 555.96 | 557.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 555.25 | 555.82 | 556.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 552.00 | 556.52 | 556.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 552.80 | 555.78 | 556.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:00:00 | 552.60 | 554.23 | 555.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 558.25 | 551.95 | 551.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 558.25 | 551.95 | 551.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 565.60 | 557.22 | 554.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 558.50 | 558.88 | 555.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:30:00 | 560.85 | 558.88 | 555.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 557.90 | 558.07 | 556.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 550.25 | 558.07 | 556.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 552.80 | 557.02 | 555.86 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 541.00 | 553.32 | 554.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 540.30 | 546.50 | 550.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 542.20 | 535.80 | 538.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 547.00 | 540.36 | 540.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 547.00 | 540.36 | 540.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 548.70 | 542.03 | 540.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 563.80 | 566.09 | 557.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:30:00 | 573.75 | 566.80 | 559.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 380.00 | 2024-05-23 09:15:00 | 375.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-06-25 12:45:00 | 329.00 | 2024-07-02 10:15:00 | 326.50 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-07-05 11:45:00 | 330.30 | 2024-07-10 13:15:00 | 332.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2024-07-05 12:45:00 | 330.45 | 2024-07-10 13:15:00 | 332.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-07-05 14:00:00 | 330.90 | 2024-07-10 13:15:00 | 332.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-07-05 15:00:00 | 331.20 | 2024-07-10 13:15:00 | 332.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-07-15 11:15:00 | 332.70 | 2024-07-19 09:15:00 | 316.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 12:15:00 | 332.80 | 2024-07-19 09:15:00 | 316.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 13:00:00 | 332.40 | 2024-07-19 09:15:00 | 315.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 15:00:00 | 331.05 | 2024-07-19 09:15:00 | 314.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:30:00 | 330.30 | 2024-07-19 09:15:00 | 313.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 11:15:00 | 332.70 | 2024-07-22 09:15:00 | 314.05 | STOP_HIT | 0.50 | 5.61% |
| SELL | retest2 | 2024-07-15 12:15:00 | 332.80 | 2024-07-22 09:15:00 | 314.05 | STOP_HIT | 0.50 | 5.63% |
| SELL | retest2 | 2024-07-15 13:00:00 | 332.40 | 2024-07-22 09:15:00 | 314.05 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2024-07-15 15:00:00 | 331.05 | 2024-07-22 09:15:00 | 314.05 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2024-07-16 12:30:00 | 330.30 | 2024-07-22 09:15:00 | 314.05 | STOP_HIT | 0.50 | 4.92% |
| BUY | retest2 | 2024-07-31 09:15:00 | 323.55 | 2024-08-02 09:15:00 | 311.90 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-07-31 11:00:00 | 321.80 | 2024-08-02 09:15:00 | 311.90 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-07-31 15:00:00 | 321.80 | 2024-08-02 09:15:00 | 311.90 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-08-08 09:15:00 | 293.10 | 2024-08-08 12:15:00 | 299.25 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-08-21 12:15:00 | 317.00 | 2024-08-23 12:15:00 | 315.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-22 09:15:00 | 318.30 | 2024-08-23 12:15:00 | 315.45 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-08-22 13:30:00 | 316.80 | 2024-08-23 12:15:00 | 315.45 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-08-22 15:00:00 | 319.00 | 2024-08-23 12:15:00 | 315.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-06 09:30:00 | 313.25 | 2024-09-10 12:15:00 | 315.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-09-10 09:45:00 | 313.45 | 2024-09-10 12:15:00 | 315.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-09-10 10:45:00 | 312.75 | 2024-09-10 12:15:00 | 315.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest1 | 2024-09-16 09:15:00 | 316.60 | 2024-09-16 11:15:00 | 313.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2024-09-16 10:00:00 | 316.35 | 2024-09-16 11:15:00 | 313.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2024-09-16 11:00:00 | 315.75 | 2024-09-16 11:15:00 | 313.35 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-09-26 11:30:00 | 342.50 | 2024-10-01 10:15:00 | 337.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-10 11:00:00 | 313.10 | 2024-10-14 15:15:00 | 324.20 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-10-10 12:15:00 | 313.15 | 2024-10-14 15:15:00 | 324.20 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-10-14 10:45:00 | 312.50 | 2024-10-14 15:15:00 | 324.20 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2024-11-04 13:00:00 | 294.45 | 2024-11-05 09:15:00 | 289.40 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-04 14:00:00 | 291.65 | 2024-11-05 09:15:00 | 289.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-11-04 15:00:00 | 292.30 | 2024-11-05 09:15:00 | 289.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-05 09:15:00 | 293.50 | 2024-11-05 09:15:00 | 289.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-12-11 09:15:00 | 293.05 | 2024-12-11 15:15:00 | 289.05 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-12-11 11:30:00 | 292.90 | 2024-12-11 15:15:00 | 289.05 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-12-13 09:15:00 | 285.15 | 2024-12-16 12:15:00 | 289.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-12-16 09:30:00 | 288.70 | 2024-12-16 12:15:00 | 289.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-12-16 10:45:00 | 288.70 | 2024-12-16 12:15:00 | 289.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-12-20 09:15:00 | 275.85 | 2024-12-20 10:15:00 | 280.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-20 09:45:00 | 276.15 | 2024-12-20 10:15:00 | 280.55 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-12-20 14:00:00 | 277.10 | 2024-12-20 15:15:00 | 263.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:30:00 | 276.45 | 2024-12-20 15:15:00 | 262.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:00:00 | 277.10 | 2024-12-23 12:15:00 | 273.00 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2024-12-20 14:30:00 | 276.45 | 2024-12-23 12:15:00 | 273.00 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-01-10 13:15:00 | 235.99 | 2025-01-13 14:15:00 | 224.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:15:00 | 235.99 | 2025-01-14 14:15:00 | 226.41 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2025-01-24 11:15:00 | 245.63 | 2025-01-24 14:15:00 | 242.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-29 11:15:00 | 229.68 | 2025-01-30 09:15:00 | 234.79 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-30 13:30:00 | 229.70 | 2025-01-31 10:15:00 | 235.53 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-02-07 10:30:00 | 242.25 | 2025-02-10 09:15:00 | 239.32 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-02-13 13:15:00 | 229.35 | 2025-02-14 11:15:00 | 217.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 229.35 | 2025-02-17 14:15:00 | 215.85 | STOP_HIT | 0.50 | 5.89% |
| BUY | retest1 | 2025-03-07 13:30:00 | 226.25 | 2025-03-10 11:15:00 | 220.76 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2025-03-07 14:30:00 | 226.60 | 2025-03-10 11:15:00 | 220.76 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-03-28 11:15:00 | 223.60 | 2025-04-02 14:15:00 | 226.35 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-28 12:00:00 | 223.02 | 2025-04-02 14:15:00 | 226.35 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-04-28 14:15:00 | 217.75 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-04-29 11:45:00 | 217.20 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-04-29 13:30:00 | 217.56 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-04-29 14:15:00 | 217.21 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-04-30 09:15:00 | 216.57 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-04-30 11:45:00 | 217.48 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-04-30 12:30:00 | 217.27 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-05-02 10:15:00 | 217.30 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-05-02 11:15:00 | 215.17 | 2025-05-05 15:15:00 | 217.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-05-08 13:00:00 | 210.92 | 2025-05-09 09:15:00 | 200.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 210.92 | 2025-05-09 15:15:00 | 206.00 | STOP_HIT | 0.50 | 2.33% |
| BUY | retest2 | 2025-05-16 09:15:00 | 230.85 | 2025-05-16 14:15:00 | 226.11 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-05-16 10:00:00 | 229.16 | 2025-05-16 14:15:00 | 226.11 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-19 09:30:00 | 229.05 | 2025-05-19 15:15:00 | 226.15 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-20 10:00:00 | 230.40 | 2025-05-20 14:15:00 | 224.90 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2025-05-28 09:15:00 | 256.23 | 2025-05-30 11:15:00 | 247.95 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-06-04 09:15:00 | 249.67 | 2025-06-12 13:15:00 | 254.80 | STOP_HIT | 1.00 | 2.05% |
| SELL | retest2 | 2025-07-09 12:00:00 | 263.65 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-09 13:30:00 | 263.55 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-10 10:45:00 | 263.55 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-10 11:45:00 | 263.20 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-16 09:30:00 | 261.05 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-16 14:45:00 | 260.20 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-23 09:15:00 | 278.00 | 2025-07-25 09:15:00 | 266.95 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-07-30 13:15:00 | 254.20 | 2025-07-30 14:15:00 | 261.90 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-07-31 09:30:00 | 253.20 | 2025-08-01 11:15:00 | 240.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:30:00 | 253.20 | 2025-08-04 12:15:00 | 242.00 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-08-25 12:15:00 | 237.17 | 2025-09-01 15:15:00 | 232.50 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-09-05 13:45:00 | 244.22 | 2025-09-12 11:15:00 | 267.96 | TARGET_HIT | 1.00 | 9.72% |
| BUY | retest2 | 2025-09-08 15:00:00 | 245.14 | 2025-09-12 12:15:00 | 268.64 | TARGET_HIT | 1.00 | 9.59% |
| BUY | retest2 | 2025-09-09 15:00:00 | 243.60 | 2025-09-12 12:15:00 | 269.65 | TARGET_HIT | 1.00 | 10.70% |
| BUY | retest1 | 2025-09-25 09:15:00 | 316.18 | 2025-09-26 13:15:00 | 311.53 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-29 09:15:00 | 321.26 | 2025-10-07 15:15:00 | 332.00 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2025-10-29 09:15:00 | 352.15 | 2025-10-31 10:15:00 | 341.85 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-10-30 11:45:00 | 347.25 | 2025-10-31 10:15:00 | 341.85 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-04 09:15:00 | 340.05 | 2025-11-06 10:15:00 | 323.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 340.05 | 2025-11-07 09:15:00 | 326.35 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-11-10 09:45:00 | 339.55 | 2025-11-10 10:15:00 | 341.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-13 09:30:00 | 345.75 | 2025-11-13 13:15:00 | 340.15 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-17 14:30:00 | 337.65 | 2025-11-21 09:15:00 | 320.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 337.45 | 2025-11-21 09:15:00 | 320.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:30:00 | 337.65 | 2025-11-25 09:15:00 | 322.00 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-11-17 15:15:00 | 337.45 | 2025-11-25 09:15:00 | 322.00 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest2 | 2025-11-28 10:45:00 | 328.20 | 2025-12-04 10:15:00 | 361.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-01 09:15:00 | 336.30 | 2025-12-05 09:15:00 | 369.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-14 09:15:00 | 552.25 | 2026-01-19 14:15:00 | 548.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-05 09:15:00 | 593.00 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-02-05 15:00:00 | 605.70 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-02-06 09:15:00 | 591.65 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-02-18 12:15:00 | 565.20 | 2026-02-18 12:15:00 | 575.40 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-19 15:00:00 | 562.85 | 2026-02-23 09:15:00 | 574.30 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-23 10:45:00 | 565.50 | 2026-02-25 09:15:00 | 574.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2026-02-26 14:45:00 | 583.85 | 2026-02-27 11:15:00 | 578.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2026-02-27 09:15:00 | 585.15 | 2026-02-27 11:15:00 | 578.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-04 09:15:00 | 560.80 | 2026-03-12 09:15:00 | 532.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 560.80 | 2026-03-13 09:15:00 | 504.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-07 09:45:00 | 508.25 | 2026-04-15 09:15:00 | 559.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:45:00 | 508.30 | 2026-04-15 09:15:00 | 559.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 13:30:00 | 508.40 | 2026-04-15 09:15:00 | 559.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 522.95 | 2026-04-21 12:15:00 | 554.25 | STOP_HIT | 1.00 | 5.99% |
| BUY | retest2 | 2026-04-13 10:15:00 | 523.35 | 2026-04-21 12:15:00 | 554.25 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2026-04-23 09:15:00 | 552.00 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-04-23 10:00:00 | 552.80 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-23 12:00:00 | 552.60 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-05-06 10:15:00 | 542.20 | 2026-05-06 12:15:00 | 547.00 | STOP_HIT | 1.00 | -0.89% |
