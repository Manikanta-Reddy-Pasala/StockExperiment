# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 742.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 95 |
| ALERT2 | 94 |
| ALERT2_SKIP | 52 |
| ALERT3 | 252 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 128 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 134 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 99
- **Target hits / Stop hits / Partials:** 2 / 128 / 4
- **Avg / median % per leg:** -0.39% / -0.68%
- **Sum % (uncompounded):** -52.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 19 | 25.7% | 2 | 72 | 0 | -0.29% | -21.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| BUY @ 3rd Alert (retest2) | 73 | 19 | 26.0% | 2 | 71 | 0 | -0.26% | -19.1% |
| SELL (all) | 60 | 16 | 26.7% | 0 | 56 | 4 | -0.51% | -30.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.62% | -1.6% |
| SELL @ 3rd Alert (retest2) | 59 | 16 | 27.1% | 0 | 55 | 4 | -0.49% | -29.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.14% | -4.3% |
| retest2 (combined) | 132 | 35 | 26.5% | 2 | 126 | 4 | -0.36% | -48.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 346.50 | 344.01 | 343.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 348.10 | 345.24 | 344.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 348.15 | 348.96 | 347.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 348.50 | 348.96 | 347.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 347.95 | 348.75 | 347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 348.15 | 348.75 | 347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 349.70 | 348.94 | 347.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 351.00 | 348.94 | 347.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 352.00 | 348.96 | 348.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 14:15:00 | 366.90 | 372.05 | 372.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 14:15:00 | 366.90 | 372.05 | 372.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 365.00 | 370.64 | 371.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 365.00 | 355.34 | 359.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 365.00 | 355.34 | 359.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 362.80 | 356.83 | 359.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 358.45 | 356.83 | 359.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 347.75 | 339.63 | 342.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 347.75 | 339.63 | 342.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 349.30 | 341.57 | 343.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 349.30 | 341.57 | 343.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 350.55 | 344.73 | 344.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 351.05 | 345.99 | 344.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 359.65 | 359.96 | 356.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 355.60 | 359.96 | 356.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 360.30 | 360.02 | 357.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 356.75 | 360.02 | 357.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 356.80 | 358.95 | 357.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 361.05 | 358.95 | 357.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 360.50 | 361.76 | 361.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:45:00 | 360.15 | 361.51 | 360.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 359.95 | 360.98 | 360.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 13:15:00 | 358.85 | 360.56 | 360.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 358.85 | 360.56 | 360.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 357.40 | 359.64 | 360.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 15:15:00 | 353.30 | 353.28 | 355.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 09:15:00 | 351.30 | 353.28 | 355.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 355.10 | 353.50 | 354.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 355.10 | 353.50 | 354.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 354.90 | 353.78 | 354.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 355.45 | 353.78 | 354.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 354.95 | 354.01 | 354.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:45:00 | 354.85 | 354.01 | 354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 352.65 | 353.74 | 354.63 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 360.95 | 355.66 | 355.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 15:15:00 | 380.20 | 364.54 | 360.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 364.70 | 374.40 | 368.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:45:00 | 366.75 | 374.40 | 368.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 364.00 | 372.32 | 368.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 354.85 | 372.32 | 368.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 355.05 | 363.56 | 364.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 352.75 | 354.61 | 357.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 349.50 | 348.11 | 351.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 11:45:00 | 350.00 | 348.11 | 351.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 350.10 | 347.71 | 349.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 350.10 | 347.71 | 349.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 351.35 | 348.44 | 349.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 352.15 | 348.44 | 349.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 350.60 | 349.31 | 349.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 349.20 | 349.48 | 349.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 15:15:00 | 341.35 | 340.84 | 340.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 341.35 | 340.84 | 340.81 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 340.05 | 340.68 | 340.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 337.25 | 339.09 | 339.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 12:15:00 | 335.10 | 335.07 | 336.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 13:00:00 | 335.10 | 335.07 | 336.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 335.65 | 335.18 | 336.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 335.65 | 335.18 | 336.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 333.00 | 334.66 | 335.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 333.45 | 334.66 | 335.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 325.90 | 324.15 | 326.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 325.90 | 324.15 | 326.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 326.55 | 324.63 | 326.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 327.20 | 324.63 | 326.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 327.15 | 325.13 | 326.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 327.15 | 325.13 | 326.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 328.00 | 325.71 | 326.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 328.35 | 325.71 | 326.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 328.05 | 326.76 | 327.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 324.40 | 327.37 | 327.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 328.65 | 327.62 | 327.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 328.65 | 327.62 | 327.60 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 325.50 | 327.41 | 327.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 15:15:00 | 324.05 | 326.74 | 327.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 323.00 | 322.73 | 324.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 323.80 | 322.67 | 324.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 326.00 | 323.40 | 324.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 326.45 | 323.40 | 324.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 328.10 | 324.34 | 324.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 328.10 | 324.34 | 324.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 330.40 | 325.83 | 325.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 332.65 | 329.79 | 327.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 336.20 | 338.54 | 334.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 336.20 | 338.54 | 334.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 369.20 | 370.37 | 364.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 10:15:00 | 371.80 | 370.37 | 364.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 377.95 | 367.92 | 365.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-09 13:15:00 | 408.98 | 401.53 | 395.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 383.30 | 395.30 | 396.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 382.80 | 392.80 | 394.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 13:15:00 | 390.95 | 390.79 | 393.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:45:00 | 390.65 | 390.79 | 393.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 387.40 | 389.90 | 392.17 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 393.00 | 390.90 | 390.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 396.00 | 393.09 | 391.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 391.50 | 392.97 | 392.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 394.00 | 392.97 | 392.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 390.00 | 392.38 | 391.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 390.00 | 392.38 | 391.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 389.55 | 391.81 | 391.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 389.20 | 391.81 | 391.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 392.65 | 391.85 | 391.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 393.45 | 392.08 | 391.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 394.90 | 392.08 | 391.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 402.25 | 406.07 | 406.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 402.25 | 406.07 | 406.46 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 407.70 | 406.34 | 406.34 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 406.15 | 406.31 | 406.32 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 406.70 | 406.38 | 406.35 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 404.55 | 406.21 | 406.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 404.00 | 405.77 | 406.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 405.75 | 403.95 | 404.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 406.50 | 403.95 | 404.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 406.95 | 404.55 | 405.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 406.95 | 404.55 | 405.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 407.15 | 405.07 | 405.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 12:45:00 | 403.35 | 404.68 | 405.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 405.90 | 403.55 | 404.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 11:15:00 | 407.15 | 404.86 | 404.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 407.15 | 404.86 | 404.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 12:15:00 | 412.45 | 408.03 | 406.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 11:15:00 | 409.00 | 410.11 | 408.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-09 12:00:00 | 409.00 | 410.11 | 408.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 406.00 | 409.28 | 408.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 406.00 | 409.28 | 408.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 407.40 | 408.91 | 408.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 405.65 | 408.91 | 408.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 409.40 | 408.76 | 408.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 407.55 | 408.53 | 408.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 413.00 | 409.42 | 408.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:30:00 | 413.30 | 410.63 | 409.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 13:15:00 | 413.20 | 410.63 | 409.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 407.70 | 409.31 | 409.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 407.70 | 409.31 | 409.31 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 409.40 | 409.33 | 409.32 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 407.75 | 409.01 | 409.18 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 414.50 | 410.11 | 409.65 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 409.30 | 414.54 | 414.73 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 11:15:00 | 416.55 | 414.94 | 414.89 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 413.75 | 414.80 | 414.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 411.65 | 414.17 | 414.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 418.35 | 413.53 | 414.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 416.95 | 413.53 | 414.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 420.80 | 414.98 | 414.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 423.10 | 417.64 | 415.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 414.95 | 417.10 | 415.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 414.95 | 417.10 | 415.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 414.20 | 416.52 | 415.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 412.50 | 416.52 | 415.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 413.40 | 415.90 | 415.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 413.40 | 415.90 | 415.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 416.40 | 416.00 | 415.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 420.60 | 416.47 | 415.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:30:00 | 419.00 | 421.19 | 420.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 418.70 | 420.56 | 420.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 418.70 | 420.56 | 420.62 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 420.80 | 420.64 | 420.62 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 15:15:00 | 419.55 | 420.42 | 420.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 415.95 | 419.53 | 420.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 416.15 | 412.39 | 414.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 416.15 | 412.39 | 414.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 412.50 | 412.41 | 414.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 409.60 | 411.93 | 413.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:15:00 | 409.85 | 411.93 | 413.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 417.00 | 414.56 | 414.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 417.00 | 414.56 | 414.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 09:15:00 | 420.30 | 417.09 | 415.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 418.30 | 419.71 | 418.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 412.55 | 419.71 | 418.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 414.70 | 418.71 | 417.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 414.70 | 418.71 | 417.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 414.95 | 417.96 | 417.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 414.05 | 417.96 | 417.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 411.60 | 416.69 | 416.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 407.40 | 413.06 | 414.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 411.95 | 408.75 | 411.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 412.70 | 408.75 | 411.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 414.50 | 409.90 | 411.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 414.50 | 409.90 | 411.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 415.55 | 411.03 | 412.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 415.55 | 411.03 | 412.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 418.25 | 413.45 | 413.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 420.00 | 415.47 | 414.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 421.00 | 421.78 | 419.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:30:00 | 421.15 | 421.78 | 419.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 418.05 | 420.88 | 419.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 418.05 | 420.88 | 419.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 417.20 | 420.14 | 419.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 416.75 | 420.14 | 419.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 418.65 | 419.64 | 419.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 418.40 | 419.64 | 419.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 417.50 | 419.21 | 419.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 418.20 | 419.21 | 419.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 416.45 | 418.66 | 418.79 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 419.75 | 418.96 | 418.86 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 416.75 | 419.03 | 419.06 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 424.50 | 419.40 | 419.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 429.00 | 423.99 | 421.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 424.95 | 426.22 | 424.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 424.95 | 426.22 | 424.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 423.95 | 425.77 | 424.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 423.50 | 425.77 | 424.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 424.90 | 425.59 | 424.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:15:00 | 423.15 | 425.59 | 424.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 422.35 | 424.94 | 424.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 420.40 | 424.94 | 424.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 419.75 | 423.91 | 423.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 419.75 | 423.91 | 423.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 419.80 | 423.08 | 423.52 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 428.75 | 423.69 | 423.07 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 417.70 | 422.79 | 422.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 415.70 | 421.37 | 422.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 446.35 | 414.29 | 414.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:30:00 | 451.20 | 414.29 | 414.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 10:15:00 | 434.95 | 418.43 | 416.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 09:15:00 | 453.05 | 437.90 | 428.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 09:15:00 | 439.95 | 443.40 | 436.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 439.95 | 443.40 | 436.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 438.00 | 440.76 | 437.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 438.00 | 440.76 | 437.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 439.15 | 440.44 | 437.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 438.00 | 440.44 | 437.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 435.55 | 439.46 | 437.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:45:00 | 435.30 | 439.46 | 437.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 431.95 | 437.96 | 437.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 432.00 | 437.96 | 437.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 432.00 | 435.79 | 436.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 14:15:00 | 429.50 | 433.56 | 435.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 433.45 | 433.08 | 434.51 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 13:15:00 | 437.00 | 435.30 | 435.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 443.45 | 437.59 | 436.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 440.15 | 442.23 | 440.22 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 438.50 | 439.57 | 439.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 15:15:00 | 435.00 | 438.29 | 439.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 439.40 | 438.41 | 438.96 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 442.80 | 439.69 | 439.44 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 435.40 | 438.97 | 439.38 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 439.75 | 438.44 | 438.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 443.00 | 439.45 | 438.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 437.85 | 440.96 | 440.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 437.85 | 440.96 | 440.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 437.50 | 440.27 | 439.80 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 432.55 | 438.72 | 439.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 430.00 | 431.89 | 433.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 439.10 | 433.33 | 434.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 439.10 | 433.33 | 434.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 437.80 | 434.22 | 434.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 439.20 | 434.22 | 434.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 436.80 | 435.06 | 434.90 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 433.40 | 434.75 | 434.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 428.10 | 433.42 | 434.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 433.55 | 432.68 | 433.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:00:00 | 433.55 | 432.68 | 433.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 434.80 | 433.11 | 433.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 434.80 | 433.11 | 433.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 429.10 | 432.31 | 433.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 15:00:00 | 428.90 | 431.62 | 432.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 428.85 | 430.57 | 432.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 439.80 | 433.84 | 433.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 439.80 | 433.84 | 433.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 454.10 | 442.23 | 439.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 501.75 | 502.42 | 490.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 501.75 | 502.42 | 490.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 489.10 | 496.68 | 491.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 491.00 | 496.68 | 491.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 483.35 | 494.02 | 490.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 483.35 | 494.02 | 490.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 482.75 | 488.92 | 489.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 09:15:00 | 481.50 | 486.44 | 488.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 14:15:00 | 485.35 | 484.40 | 486.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 485.35 | 484.40 | 486.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 494.90 | 486.44 | 486.84 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 489.90 | 487.66 | 487.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 15:15:00 | 492.25 | 489.70 | 488.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 11:15:00 | 490.15 | 490.28 | 489.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:00:00 | 490.15 | 490.28 | 489.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 489.30 | 490.09 | 489.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:00:00 | 489.30 | 490.09 | 489.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 491.45 | 490.36 | 489.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:30:00 | 490.15 | 490.36 | 489.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 490.50 | 490.61 | 489.88 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 485.70 | 489.41 | 489.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 484.15 | 486.80 | 487.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 489.90 | 486.84 | 487.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 490.60 | 486.84 | 487.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 489.00 | 487.27 | 487.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 487.50 | 487.27 | 487.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 486.85 | 487.19 | 487.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 484.75 | 486.57 | 487.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:30:00 | 485.05 | 486.10 | 487.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:00:00 | 484.20 | 486.10 | 487.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 484.90 | 480.34 | 481.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 485.95 | 481.46 | 481.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 484.15 | 482.00 | 481.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 484.15 | 482.00 | 481.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 15:15:00 | 489.70 | 484.76 | 483.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 12:15:00 | 491.25 | 492.30 | 489.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 491.25 | 492.30 | 489.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 490.25 | 491.89 | 489.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 490.35 | 491.89 | 489.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 492.55 | 491.95 | 490.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 500.55 | 491.95 | 490.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 485.80 | 494.12 | 492.76 | SL hit (close<static) qty=1.00 sl=489.90 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 520.60 | 523.21 | 523.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 516.75 | 521.10 | 522.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 497.25 | 496.61 | 502.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 497.25 | 496.61 | 502.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 500.45 | 497.96 | 501.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 500.45 | 497.96 | 501.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 497.35 | 497.91 | 500.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 491.95 | 496.49 | 499.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 492.45 | 495.68 | 499.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 492.30 | 496.65 | 497.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 492.80 | 496.05 | 497.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 496.30 | 495.73 | 496.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 496.30 | 495.73 | 496.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 499.20 | 496.43 | 496.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 497.70 | 496.43 | 496.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 499.25 | 496.99 | 497.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 495.75 | 496.99 | 497.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 494.80 | 496.55 | 496.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 497.90 | 496.83 | 496.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 10:15:00 | 497.90 | 496.83 | 496.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 14:15:00 | 499.30 | 498.04 | 497.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 497.00 | 497.83 | 497.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 491.90 | 497.83 | 497.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 496.35 | 497.54 | 497.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 494.25 | 497.54 | 497.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 498.50 | 497.73 | 497.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 13:00:00 | 500.00 | 498.21 | 497.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:00:00 | 500.00 | 498.69 | 498.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:45:00 | 499.90 | 499.02 | 498.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 499.85 | 499.06 | 498.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 500.00 | 499.32 | 498.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 499.45 | 499.32 | 498.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 498.70 | 499.20 | 498.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 497.70 | 499.20 | 498.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 498.05 | 498.97 | 498.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 498.05 | 498.97 | 498.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 496.25 | 498.42 | 498.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 496.25 | 498.42 | 498.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 495.20 | 497.78 | 498.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 488.00 | 485.99 | 490.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 488.00 | 485.99 | 490.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 488.65 | 486.52 | 490.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 481.00 | 486.52 | 490.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 11:00:00 | 485.25 | 485.58 | 489.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 486.00 | 482.31 | 481.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 486.00 | 482.31 | 481.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 491.00 | 484.04 | 482.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 469.15 | 485.37 | 484.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:45:00 | 466.05 | 485.37 | 484.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 474.30 | 483.15 | 483.90 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 486.00 | 478.17 | 477.36 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 475.75 | 478.27 | 478.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 473.35 | 477.28 | 478.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 433.75 | 432.43 | 441.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 437.00 | 432.43 | 441.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 442.20 | 435.51 | 441.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 442.20 | 435.51 | 441.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 437.80 | 435.97 | 441.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 442.40 | 435.97 | 441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 411.50 | 406.60 | 410.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 411.50 | 406.60 | 410.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 416.00 | 408.48 | 411.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:00:00 | 416.00 | 408.48 | 411.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 416.20 | 410.02 | 411.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 416.80 | 410.02 | 411.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 417.00 | 413.38 | 413.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 12:15:00 | 420.00 | 416.77 | 415.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 417.15 | 417.46 | 415.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:15:00 | 426.30 | 417.46 | 415.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 417.80 | 417.53 | 416.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 417.80 | 417.53 | 416.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 418.85 | 417.80 | 416.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 415.00 | 417.30 | 416.55 | SL hit (close<ema400) qty=1.00 sl=416.55 alert=retest1 |

### Cycle 64 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 403.70 | 413.89 | 415.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 401.05 | 404.83 | 407.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 403.00 | 402.12 | 405.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 12:00:00 | 403.00 | 402.12 | 405.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 404.80 | 402.36 | 404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 404.80 | 402.36 | 404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 403.95 | 402.68 | 404.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 395.75 | 402.68 | 404.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:15:00 | 397.60 | 397.28 | 398.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 409.95 | 399.68 | 399.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 409.95 | 399.68 | 399.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 419.05 | 405.05 | 401.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 417.85 | 419.47 | 415.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 417.85 | 419.47 | 415.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 425.65 | 420.31 | 416.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 435.00 | 420.31 | 416.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 427.90 | 421.70 | 417.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:00:00 | 428.00 | 424.12 | 421.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 429.15 | 430.95 | 430.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 428.45 | 430.45 | 430.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 428.45 | 430.45 | 430.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 427.65 | 429.89 | 430.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 431.00 | 429.27 | 429.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 431.00 | 429.27 | 429.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 426.75 | 428.77 | 429.46 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 430.75 | 429.45 | 429.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 433.50 | 430.47 | 429.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 432.05 | 432.54 | 431.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 430.80 | 432.54 | 431.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 433.00 | 432.64 | 431.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 439.30 | 434.53 | 433.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 437.20 | 438.94 | 437.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 434.95 | 436.97 | 437.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 434.95 | 436.97 | 437.09 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 437.65 | 437.22 | 437.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 447.45 | 439.26 | 438.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 471.95 | 476.46 | 468.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 478.05 | 476.24 | 470.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 477.35 | 476.24 | 470.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 477.30 | 476.31 | 471.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 477.30 | 481.10 | 478.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 479.75 | 480.83 | 478.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:45:00 | 483.40 | 481.22 | 478.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 13:15:00 | 483.90 | 481.22 | 478.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:15:00 | 483.85 | 481.50 | 479.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 483.00 | 481.80 | 479.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 464.20 | 478.48 | 478.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 464.20 | 478.48 | 478.29 | SL hit (close<static) qty=1.00 sl=465.05 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 474.85 | 477.75 | 477.97 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 486.15 | 478.82 | 478.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 496.00 | 484.15 | 481.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 494.75 | 498.08 | 493.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:30:00 | 494.00 | 498.08 | 493.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 12:15:00 | 493.05 | 497.07 | 493.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:45:00 | 492.15 | 497.07 | 493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 491.90 | 496.04 | 493.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:15:00 | 491.50 | 496.04 | 493.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 490.35 | 494.90 | 493.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:30:00 | 490.85 | 494.90 | 493.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 497.85 | 495.14 | 493.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 501.20 | 496.35 | 494.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 501.90 | 498.12 | 495.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:30:00 | 500.65 | 499.24 | 497.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 502.00 | 499.61 | 497.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 507.15 | 506.26 | 503.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 508.20 | 506.26 | 503.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:00:00 | 507.85 | 507.34 | 504.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 511.85 | 507.93 | 505.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 502.65 | 508.68 | 507.54 | SL hit (close<static) qty=1.00 sl=503.30 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 495.65 | 506.07 | 506.46 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 510.50 | 507.25 | 506.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 512.50 | 508.30 | 507.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 513.15 | 513.15 | 511.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:00:00 | 513.15 | 513.15 | 511.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 512.80 | 513.08 | 511.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 509.90 | 513.08 | 511.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 512.00 | 512.86 | 511.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 506.75 | 512.86 | 511.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 503.10 | 510.91 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 503.10 | 510.91 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 508.25 | 510.38 | 510.41 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 510.95 | 509.51 | 509.41 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 507.25 | 509.10 | 509.25 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 517.00 | 510.68 | 509.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 521.20 | 514.70 | 512.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 512.55 | 519.26 | 517.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 524.15 | 520.24 | 518.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 15:15:00 | 576.57 | 562.61 | 553.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 577.45 | 580.33 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 567.00 | 577.66 | 579.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 569.00 | 561.77 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 567.70 | 562.96 | 567.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 562.70 | 566.52 | 567.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 560.65 | 564.81 | 566.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 580.20 | 559.84 | 555.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 582.35 | 583.11 | 576.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 582.35 | 583.11 | 576.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 577.50 | 581.14 | 577.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 577.50 | 581.14 | 577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 577.50 | 580.41 | 577.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 573.40 | 580.41 | 577.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 572.00 | 578.73 | 577.05 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 569.00 | 575.67 | 575.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 565.80 | 569.34 | 571.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 571.00 | 569.67 | 571.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 571.55 | 570.05 | 571.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 571.55 | 570.05 | 571.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 574.10 | 570.86 | 571.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 570.30 | 570.75 | 571.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 569.80 | 570.56 | 571.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 561.50 | 570.41 | 571.30 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 580.00 | 570.24 | 569.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 589.30 | 574.05 | 571.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 572.95 | 581.03 | 577.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 571.20 | 579.07 | 576.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 571.20 | 579.07 | 576.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 566.35 | 574.35 | 574.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 563.60 | 567.92 | 569.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 562.40 | 562.28 | 565.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:15:00 | 563.70 | 562.28 | 565.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 569.85 | 563.79 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 570.65 | 563.79 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 569.90 | 565.01 | 565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 571.20 | 565.01 | 565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 570.25 | 566.78 | 566.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 573.25 | 568.07 | 567.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:30:00 | 579.90 | 574.47 | 571.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 592.60 | 590.48 | 590.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 600.20 | 592.43 | 591.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 592.40 | 594.23 | 592.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 594.00 | 594.18 | 592.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 620.15 | 594.18 | 592.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 612.50 | 618.35 | 620.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 605.40 | 603.83 | 608.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 12:00:00 | 605.40 | 603.83 | 608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 609.20 | 605.39 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 608.50 | 605.39 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 608.55 | 606.03 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 610.00 | 606.03 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 605.50 | 605.92 | 608.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 601.90 | 605.92 | 608.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 604.25 | 605.59 | 607.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 597.80 | 602.10 | 604.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 598.50 | 601.63 | 603.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 598.85 | 600.69 | 602.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:00:00 | 597.30 | 599.88 | 601.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 593.75 | 596.16 | 598.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 592.10 | 596.16 | 598.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 593.65 | 595.37 | 597.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 587.25 | 595.00 | 596.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 600.60 | 597.97 | 596.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 594.95 | 598.94 | 596.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 595.65 | 598.28 | 596.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:15:00 | 595.30 | 598.28 | 596.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 598.40 | 598.30 | 597.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:30:00 | 595.95 | 598.30 | 597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 593.95 | 597.46 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 594.60 | 597.46 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 596.90 | 597.35 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 598.00 | 597.35 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 598.30 | 597.54 | 596.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 599.45 | 597.54 | 596.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 599.40 | 598.03 | 597.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 599.55 | 598.34 | 597.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 595.40 | 597.75 | 597.28 | SL hit (close<static) qty=1.00 sl=596.05 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 593.45 | 596.89 | 596.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 590.40 | 595.59 | 596.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 583.70 | 580.39 | 584.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 581.60 | 580.63 | 584.36 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 591.80 | 585.97 | 585.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 594.15 | 587.60 | 586.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 596.40 | 601.82 | 597.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 597.05 | 600.87 | 597.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:15:00 | 596.40 | 600.87 | 597.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 593.00 | 599.30 | 597.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 593.00 | 599.30 | 597.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 596.15 | 598.67 | 597.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 598.15 | 597.11 | 596.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 599.60 | 594.72 | 594.11 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 592.70 | 594.72 | 594.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 590.75 | 593.65 | 594.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 594.50 | 593.53 | 594.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 596.00 | 594.02 | 594.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 590.00 | 594.02 | 594.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 590.00 | 593.22 | 593.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 588.00 | 591.75 | 593.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 587.00 | 585.29 | 587.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 600.55 | 592.09 | 589.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 608.20 | 608.57 | 603.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 608.20 | 608.57 | 603.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 606.40 | 607.59 | 604.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 612.40 | 607.59 | 604.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 616.30 | 615.46 | 615.42 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 610.25 | 614.42 | 614.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 606.05 | 612.75 | 614.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 605.70 | 605.79 | 608.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 608.40 | 606.31 | 608.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 604.95 | 606.53 | 608.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 601.15 | 604.79 | 606.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 605.15 | 605.16 | 606.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 574.70 | 597.43 | 600.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 597.55 | 597.43 | 600.49 | SL hit (close>static) qty=0.50 sl=597.43 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 611.60 | 603.28 | 602.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 618.75 | 609.70 | 606.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 617.30 | 618.23 | 612.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:30:00 | 617.00 | 618.23 | 612.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 635.30 | 637.68 | 630.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 630.90 | 637.68 | 630.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 637.95 | 638.09 | 632.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 645.00 | 638.90 | 634.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 642.55 | 639.14 | 635.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 642.55 | 639.26 | 637.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 642.25 | 641.70 | 640.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 640.35 | 641.43 | 640.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 640.00 | 639.83 | 639.82 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 632.00 | 638.26 | 639.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 627.60 | 635.14 | 637.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 632.35 | 631.91 | 635.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 632.35 | 631.91 | 635.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 613.90 | 613.83 | 617.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 613.00 | 613.83 | 617.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 617.20 | 614.05 | 616.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 621.45 | 614.05 | 616.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 629.00 | 617.04 | 617.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 627.70 | 617.04 | 617.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 631.40 | 619.91 | 618.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 637.80 | 626.88 | 622.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 646.95 | 647.08 | 638.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 647.15 | 647.08 | 638.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 639.15 | 644.49 | 638.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 639.15 | 644.49 | 638.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 636.30 | 642.86 | 638.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 635.95 | 642.86 | 638.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 637.35 | 641.75 | 638.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 639.60 | 639.99 | 638.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 641.80 | 640.29 | 638.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 635.00 | 638.87 | 638.25 | SL hit (close<static) qty=1.00 sl=636.10 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 633.70 | 637.83 | 637.84 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 646.65 | 637.59 | 637.30 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 629.30 | 636.47 | 637.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 623.65 | 633.91 | 636.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 640.80 | 631.63 | 633.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 629.00 | 631.11 | 633.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 626.70 | 631.11 | 633.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 627.70 | 629.98 | 632.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 626.65 | 628.82 | 631.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 627.40 | 627.03 | 629.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 626.00 | 626.83 | 628.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 639.70 | 632.14 | 630.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 663.55 | 667.57 | 657.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 663.55 | 667.57 | 657.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 658.80 | 665.10 | 658.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 667.25 | 665.10 | 658.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 695.35 | 692.55 | 692.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 699.00 | 693.84 | 692.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 696.45 | 697.05 | 695.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:45:00 | 696.95 | 697.05 | 695.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 700.70 | 717.61 | 715.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 700.70 | 717.61 | 715.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 701.15 | 714.32 | 714.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 700.25 | 714.32 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 703.60 | 712.17 | 713.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 696.25 | 703.24 | 707.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 699.90 | 699.37 | 702.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 699.90 | 699.37 | 702.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 696.15 | 698.31 | 701.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 693.95 | 698.72 | 700.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 694.75 | 697.58 | 699.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 678.85 | 697.75 | 699.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 691.20 | 686.36 | 690.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 692.00 | 687.49 | 690.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 692.50 | 687.49 | 690.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 683.55 | 686.70 | 689.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 681.85 | 686.70 | 689.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 680.10 | 682.79 | 686.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 12:15:00 | 697.50 | 690.20 | 688.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 686.65 | 690.39 | 688.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 689.00 | 690.11 | 688.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 679.45 | 690.11 | 688.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 681.45 | 688.38 | 688.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:45:00 | 694.15 | 689.39 | 688.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 673.25 | 683.40 | 686.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 15:15:00 | 682.80 | 682.21 | 684.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:15:00 | 672.35 | 682.21 | 684.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 679.55 | 679.01 | 681.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 679.55 | 679.01 | 681.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | SL hit (close>ema400) qty=1.00 sl=681.54 alert=retest1 |

### Cycle 113 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 685.00 | 682.66 | 682.58 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 678.00 | 681.69 | 682.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 675.20 | 680.39 | 681.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 681.90 | 680.69 | 681.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 678.40 | 680.24 | 681.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 675.30 | 678.24 | 680.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 676.80 | 677.66 | 679.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 676.05 | 677.87 | 679.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 676.40 | 677.89 | 679.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 677.40 | 677.79 | 678.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 678.20 | 677.79 | 678.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 678.35 | 677.90 | 678.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 678.80 | 677.90 | 678.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 678.10 | 677.94 | 678.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 673.35 | 677.94 | 678.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 674.65 | 677.00 | 678.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 675.90 | 676.19 | 677.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 680.50 | 677.05 | 677.87 | SL hit (close>static) qty=1.00 sl=679.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 697.25 | 681.93 | 679.94 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 672.70 | 680.74 | 681.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 669.00 | 676.98 | 679.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 671.80 | 671.58 | 676.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 671.80 | 671.58 | 676.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 669.10 | 659.83 | 664.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 669.10 | 659.83 | 664.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 669.50 | 661.77 | 665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 669.50 | 661.77 | 665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 656.80 | 664.17 | 665.40 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 665.00 | 662.76 | 662.69 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 661.85 | 662.58 | 662.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 660.60 | 662.18 | 662.43 | Break + close below crossover candle low |

### Cycle 119 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 664.40 | 662.62 | 662.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 668.85 | 663.87 | 663.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 660.35 | 662.66 | 662.90 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 666.10 | 662.98 | 662.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 669.80 | 664.68 | 663.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 663.05 | 664.96 | 664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 665.15 | 665.00 | 664.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 666.85 | 666.00 | 664.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 658.10 | 665.20 | 664.75 | SL hit (close<static) qty=1.00 sl=662.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 660.10 | 664.18 | 664.33 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 667.50 | 664.03 | 664.02 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 660.65 | 663.35 | 663.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 659.95 | 662.67 | 663.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 657.45 | 656.01 | 658.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 657.45 | 656.01 | 658.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 650.75 | 654.96 | 657.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 648.80 | 653.03 | 656.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 616.36 | 623.00 | 633.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 622.00 | 621.27 | 628.47 | SL hit (close>ema200) qty=0.50 sl=621.27 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 610.70 | 599.73 | 598.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 616.05 | 604.89 | 601.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 607.10 | 612.59 | 607.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 608.25 | 611.72 | 607.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 613.30 | 611.95 | 608.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 614.00 | 612.48 | 608.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 605.85 | 610.73 | 608.87 | SL hit (close<static) qty=1.00 sl=606.55 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 600.00 | 607.71 | 607.77 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 615.85 | 608.10 | 607.86 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 604.55 | 610.02 | 610.19 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 612.00 | 609.51 | 609.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 614.60 | 610.53 | 609.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 618.00 | 620.44 | 617.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 618.00 | 620.44 | 617.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 618.05 | 619.96 | 617.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 616.40 | 619.96 | 617.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 620.60 | 620.09 | 617.85 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 615.35 | 617.36 | 617.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 613.30 | 615.89 | 616.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 609.65 | 607.60 | 607.60 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 606.90 | 607.61 | 607.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 600.60 | 605.83 | 606.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 560.30 | 555.84 | 566.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 559.25 | 555.84 | 566.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 568.25 | 558.99 | 566.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 568.25 | 558.99 | 566.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 567.20 | 560.63 | 566.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 571.75 | 560.63 | 566.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 572.60 | 564.52 | 567.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 572.60 | 564.52 | 567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 570.35 | 565.68 | 567.60 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 574.90 | 569.69 | 569.07 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 565.05 | 568.33 | 568.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 561.30 | 566.93 | 567.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 557.75 | 556.48 | 559.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 557.75 | 556.48 | 559.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 548.30 | 542.91 | 547.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 548.30 | 542.91 | 547.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 545.55 | 543.44 | 546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:15:00 | 550.25 | 543.44 | 546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 550.00 | 544.75 | 547.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 550.95 | 544.75 | 547.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 551.95 | 546.19 | 547.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 551.95 | 546.19 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 558.50 | 550.08 | 549.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 566.55 | 553.38 | 550.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 563.50 | 557.51 | 556.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 15:15:00 | 544.55 | 552.51 | 554.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 538.30 | 536.81 | 541.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 538.30 | 536.81 | 541.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 549.80 | 539.36 | 541.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 552.95 | 539.36 | 541.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 558.00 | 543.09 | 543.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 558.00 | 543.09 | 543.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 560.00 | 546.47 | 544.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 562.00 | 551.75 | 547.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 592.45 | 599.52 | 593.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 597.10 | 599.04 | 593.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 607.85 | 601.64 | 597.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 608.00 | 604.28 | 598.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 636.40 | 643.11 | 644.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:45:00 | 645.45 | 643.18 | 644.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 649.80 | 644.50 | 644.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:45:00 | 648.35 | 644.50 | 644.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 13:15:00 | 646.00 | 644.80 | 644.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 656.05 | 648.49 | 646.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 643.45 | 665.56 | 661.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 643.70 | 661.19 | 659.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 637.35 | 661.19 | 659.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 660.95 | 659.83 | 659.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 665.05 | 660.07 | 659.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 667.60 | 662.57 | 660.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 651.00 | 667.48 | 672.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 670.10 | 655.14 | 661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 672.25 | 658.57 | 662.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 672.25 | 658.57 | 662.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 664.65 | 660.64 | 662.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 665.45 | 660.64 | 662.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 671.05 | 662.72 | 663.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 671.05 | 662.72 | 663.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 668.90 | 663.96 | 663.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 672.70 | 665.70 | 664.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 13:15:00 | 671.85 | 672.21 | 668.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 674.85 | 672.21 | 668.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 664.00 | 670.57 | 668.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 664.00 | 670.57 | 668.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 659.00 | 668.26 | 667.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 645.95 | 668.26 | 667.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 644.70 | 663.54 | 665.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 639.45 | 658.73 | 663.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 623.20 | 620.95 | 630.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 635.05 | 620.95 | 630.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.25 | 622.81 | 630.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 624.80 | 624.65 | 630.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 638.95 | 627.51 | 631.00 | SL hit (close>static) qty=1.00 sl=636.90 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 641.60 | 634.11 | 633.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 650.80 | 637.45 | 634.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 651.00 | 656.84 | 650.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 650.80 | 655.63 | 650.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 645.65 | 655.63 | 650.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 652.55 | 655.01 | 650.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 653.00 | 655.01 | 650.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 670.50 | 665.95 | 665.37 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 657.10 | 664.70 | 665.21 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 670.00 | 663.71 | 663.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 673.55 | 667.23 | 665.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 668.00 | 669.77 | 667.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 671.00 | 670.01 | 668.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 671.65 | 670.22 | 668.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 671.80 | 670.55 | 668.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:00:00 | 671.85 | 670.55 | 668.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 672.10 | 672.38 | 669.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 672.85 | 676.17 | 672.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 679.45 | 676.45 | 673.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 678.65 | 672.20 | 671.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 681.60 | 674.08 | 672.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 676.00 | 679.51 | 676.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 678.40 | 679.29 | 676.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 678.15 | 679.29 | 676.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 679.85 | 679.10 | 677.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:15:00 | 672.90 | 679.10 | 677.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 676.00 | 678.48 | 677.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:30:00 | 673.55 | 678.48 | 677.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 675.70 | 677.92 | 677.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 677.85 | 677.92 | 677.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 678.30 | 683.60 | 682.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 687.35 | 682.79 | 682.23 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 677.60 | 681.88 | 682.35 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 684.00 | 681.96 | 681.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 685.65 | 682.69 | 682.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 714.90 | 714.95 | 707.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:30:00 | 715.00 | 714.95 | 707.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 706.20 | 713.20 | 707.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 706.20 | 713.20 | 707.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 705.35 | 711.63 | 707.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 705.35 | 711.63 | 707.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 705.65 | 710.43 | 707.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 705.65 | 710.43 | 707.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 698.55 | 707.35 | 706.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 698.60 | 707.35 | 706.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 691.15 | 704.11 | 704.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 685.45 | 696.16 | 699.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 703.00 | 695.08 | 698.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 707.00 | 697.47 | 698.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 750.20 | 697.47 | 698.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 754.80 | 708.93 | 703.96 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 15:15:00 | 351.00 | 2024-05-29 14:15:00 | 366.90 | STOP_HIT | 1.00 | 4.53% |
| BUY | retest2 | 2024-05-21 09:15:00 | 352.00 | 2024-05-29 14:15:00 | 366.90 | STOP_HIT | 1.00 | 4.23% |
| BUY | retest2 | 2024-06-12 09:15:00 | 361.05 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-14 09:45:00 | 360.50 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-06-14 10:45:00 | 360.15 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-06-14 12:45:00 | 359.95 | 2024-06-14 13:15:00 | 358.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-02 09:30:00 | 349.20 | 2024-07-10 15:15:00 | 341.35 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2024-07-23 12:15:00 | 324.40 | 2024-07-23 12:15:00 | 328.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-08-05 10:15:00 | 371.80 | 2024-08-09 13:15:00 | 408.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 377.95 | 2024-08-12 15:15:00 | 383.30 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-08-20 14:45:00 | 393.45 | 2024-08-29 14:15:00 | 402.25 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2024-08-20 15:15:00 | 394.90 | 2024-08-29 14:15:00 | 402.25 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2024-09-03 12:45:00 | 403.35 | 2024-09-04 11:15:00 | 407.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-04 10:15:00 | 405.90 | 2024-09-04 11:15:00 | 407.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-10 12:30:00 | 413.30 | 2024-09-11 12:15:00 | 407.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-09-10 13:15:00 | 413.20 | 2024-09-11 12:15:00 | 407.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-20 09:15:00 | 420.60 | 2024-09-25 10:15:00 | 418.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-09-24 12:30:00 | 419.00 | 2024-09-25 10:15:00 | 418.70 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-09-30 09:30:00 | 409.60 | 2024-10-01 10:15:00 | 417.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-09-30 10:15:00 | 409.85 | 2024-10-01 10:15:00 | 417.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-21 15:00:00 | 428.90 | 2024-11-25 10:15:00 | 439.80 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-11-22 10:15:00 | 428.85 | 2024-11-25 10:15:00 | 439.80 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-12-12 12:00:00 | 484.75 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-12-12 12:30:00 | 485.05 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-12-12 13:00:00 | 484.20 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-12-17 10:15:00 | 484.90 | 2024-12-17 11:15:00 | 484.15 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-12-20 09:15:00 | 500.55 | 2024-12-20 14:15:00 | 485.80 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-12-23 09:15:00 | 494.10 | 2025-01-08 12:15:00 | 520.60 | STOP_HIT | 1.00 | 5.36% |
| SELL | retest2 | 2025-01-15 11:30:00 | 491.95 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-01-15 13:00:00 | 492.45 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-17 09:30:00 | 492.30 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-17 10:45:00 | 492.80 | 2025-01-21 10:15:00 | 497.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-01-22 13:00:00 | 500.00 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-01-23 10:00:00 | 500.00 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-01-23 11:45:00 | 499.90 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-23 12:45:00 | 499.85 | 2025-01-24 11:15:00 | 496.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-01-28 09:15:00 | 481.00 | 2025-01-30 15:15:00 | 486.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-01-28 11:00:00 | 485.25 | 2025-01-30 15:15:00 | 486.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-02-21 09:15:00 | 426.30 | 2025-02-21 14:15:00 | 415.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-03-03 09:15:00 | 395.75 | 2025-03-05 09:15:00 | 409.95 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-03-04 13:15:00 | 397.60 | 2025-03-05 09:15:00 | 409.95 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-03-10 10:15:00 | 435.00 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-03-10 10:45:00 | 427.90 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-03-11 12:00:00 | 428.00 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-03-17 13:00:00 | 429.15 | 2025-03-17 13:15:00 | 428.45 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-03-25 09:30:00 | 439.30 | 2025-03-27 11:15:00 | 434.95 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-03-27 09:15:00 | 437.20 | 2025-03-27 11:15:00 | 434.95 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-04-02 12:30:00 | 478.05 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-04-02 13:00:00 | 477.35 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-04-02 14:15:00 | 477.30 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-04-04 10:30:00 | 477.30 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-04-04 12:45:00 | 483.40 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-04-04 13:15:00 | 483.90 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-04-04 14:15:00 | 483.85 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-04-04 15:00:00 | 483.00 | 2025-04-07 09:15:00 | 464.20 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-04-16 12:00:00 | 501.20 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-04-17 09:15:00 | 501.90 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-04-17 12:30:00 | 500.65 | 2025-04-25 09:15:00 | 502.65 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-04-17 13:30:00 | 502.00 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-04-23 11:15:00 | 508.20 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-04-23 13:00:00 | 507.85 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-04-24 10:00:00 | 511.85 | 2025-04-25 10:15:00 | 495.65 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-05-07 11:00:00 | 524.15 | 2025-05-14 15:15:00 | 576.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-23 11:00:00 | 562.70 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-23 11:30:00 | 560.65 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-11 14:00:00 | 570.30 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-11 15:00:00 | 569.80 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-25 10:30:00 | 579.90 | 2025-07-01 15:15:00 | 588.00 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-03 09:15:00 | 620.15 | 2025-07-09 10:15:00 | 615.55 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-15 13:30:00 | 597.80 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-15 15:15:00 | 598.50 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-17 09:15:00 | 598.85 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-17 13:00:00 | 597.30 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-18 10:15:00 | 592.10 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-18 13:30:00 | 593.65 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-21 09:15:00 | 587.25 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-24 12:15:00 | 599.45 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-24 12:45:00 | 599.40 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-24 14:00:00 | 599.55 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-01 14:30:00 | 598.15 | 2025-08-04 09:15:00 | 586.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-08 11:45:00 | 588.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 15:15:00 | 587.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-18 09:15:00 | 612.40 | 2025-08-21 09:15:00 | 614.10 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 574.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 571.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 574.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.26% |
| BUY | retest2 | 2025-09-08 14:15:00 | 645.00 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-08 14:45:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-10 09:15:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-11 09:15:00 | 642.25 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-23 15:15:00 | 639.60 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-24 10:00:00 | 641.80 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-29 11:15:00 | 626.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-29 13:15:00 | 627.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-29 14:30:00 | 626.65 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-30 14:45:00 | 627.40 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-08 09:15:00 | 667.25 | 2025-10-14 15:15:00 | 685.00 | STOP_HIT | 1.00 | 2.66% |
| SELL | retest2 | 2025-10-30 10:15:00 | 693.95 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-10-30 13:45:00 | 694.75 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-31 09:15:00 | 678.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-03 10:45:00 | 691.20 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-03 13:15:00 | 681.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-04 09:30:00 | 680.10 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-07 10:45:00 | 694.15 | 2025-11-10 09:15:00 | 680.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-11-11 09:15:00 | 672.35 | 2025-11-12 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-13 14:45:00 | 675.30 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-14 09:45:00 | 676.80 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-14 10:30:00 | 676.05 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-14 12:45:00 | 676.40 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-17 09:15:00 | 673.35 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-11-17 11:15:00 | 674.65 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-17 12:45:00 | 675.90 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-12-01 13:30:00 | 666.85 | 2025-12-02 09:15:00 | 658.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 09:15:00 | 616.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 14:15:00 | 622.00 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest2 | 2025-12-29 13:00:00 | 613.30 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-29 13:45:00 | 614.00 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-03 10:45:00 | 563.50 | 2026-02-03 14:15:00 | 546.30 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-16 10:00:00 | 607.85 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2026-02-16 10:30:00 | 608.00 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2026-03-10 09:15:00 | 665.05 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-03-10 11:00:00 | 667.60 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-03-24 10:30:00 | 624.80 | 2026-03-24 11:15:00 | 638.95 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-30 12:15:00 | 653.00 | 2026-04-02 15:15:00 | 652.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 671.65 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-04-10 10:30:00 | 671.80 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-10 11:00:00 | 671.85 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-10 12:30:00 | 672.10 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-04-13 10:45:00 | 679.45 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-04-17 09:15:00 | 677.85 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-20 13:45:00 | 678.30 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.19% |
