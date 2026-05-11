# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 73 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 72 |
| PARTIAL | 10 |
| TARGET_HIT | 29 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 35
- **Target hits / Stop hits / Partials:** 29 / 44 / 10
- **Avg / median % per leg:** 3.10% / 2.19%
- **Sum % (uncompounded):** 257.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 28 | 53.8% | 28 | 24 | 0 | 4.24% | 220.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 52 | 28 | 53.8% | 28 | 24 | 0 | 4.24% | 220.6% |
| SELL (all) | 31 | 20 | 64.5% | 1 | 20 | 10 | 1.19% | 36.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| SELL @ 3rd Alert (retest2) | 30 | 20 | 66.7% | 1 | 19 | 10 | 1.34% | 40.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| retest2 (combined) | 82 | 48 | 58.5% | 29 | 43 | 10 | 3.18% | 260.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 12:15:00 | 263.15 | 258.59 | 258.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 14:15:00 | 265.55 | 258.69 | 258.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 266.20 | 266.94 | 263.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 09:45:00 | 266.65 | 266.94 | 263.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 246.85 | 266.69 | 263.74 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 11:15:00 | 249.50 | 261.16 | 261.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 14:15:00 | 246.80 | 259.98 | 260.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 12:15:00 | 257.20 | 255.59 | 257.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 12:15:00 | 257.20 | 255.59 | 257.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 257.20 | 255.59 | 257.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:45:00 | 257.80 | 255.59 | 257.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 259.10 | 255.65 | 257.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 259.10 | 255.65 | 257.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 259.00 | 255.68 | 257.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 260.65 | 255.68 | 257.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 259.10 | 255.71 | 257.93 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 270.55 | 259.83 | 259.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 273.40 | 261.08 | 260.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 264.50 | 266.97 | 263.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 264.50 | 266.97 | 263.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 264.50 | 266.97 | 263.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 264.50 | 266.97 | 263.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 262.20 | 266.87 | 263.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 262.20 | 266.87 | 263.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 263.70 | 266.84 | 263.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 10:45:00 | 266.65 | 266.78 | 263.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:15:00 | 268.85 | 266.65 | 263.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 266.40 | 266.69 | 264.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 15:00:00 | 266.70 | 266.50 | 264.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 263.65 | 266.43 | 264.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:45:00 | 264.15 | 266.43 | 264.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 263.90 | 266.41 | 264.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 10:30:00 | 269.00 | 265.98 | 264.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:00:00 | 267.50 | 265.98 | 264.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:30:00 | 267.70 | 266.01 | 264.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 14:30:00 | 267.70 | 267.55 | 265.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 265.60 | 267.53 | 265.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 09:15:00 | 269.85 | 267.53 | 265.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 15:00:00 | 268.95 | 267.61 | 265.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 269.05 | 268.14 | 265.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 10:45:00 | 268.50 | 268.27 | 265.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 264.45 | 268.21 | 265.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-28 12:15:00 | 264.45 | 268.21 | 265.97 | SL hit (close<static) qty=1.00 sl=264.90 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 311.05 | 336.91 | 336.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 306.20 | 335.64 | 336.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 13:15:00 | 328.00 | 327.39 | 331.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 14:00:00 | 328.00 | 327.39 | 331.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 331.90 | 327.43 | 331.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 15:00:00 | 331.90 | 327.43 | 331.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 332.00 | 327.48 | 331.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:15:00 | 334.65 | 327.48 | 331.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 335.20 | 327.56 | 331.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:15:00 | 336.30 | 327.56 | 331.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 338.70 | 327.67 | 331.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:00:00 | 338.70 | 327.67 | 331.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 327.00 | 329.01 | 331.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 330.20 | 329.01 | 331.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 329.75 | 329.01 | 331.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:45:00 | 331.70 | 329.01 | 331.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 329.15 | 328.96 | 331.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 09:30:00 | 331.95 | 328.96 | 331.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 326.35 | 328.12 | 331.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 316.30 | 327.99 | 330.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 332.00 | 324.90 | 328.69 | SL hit (close>static) qty=1.00 sl=331.25 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 346.35 | 323.06 | 323.01 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 319.25 | 326.62 | 326.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 11:15:00 | 316.70 | 326.45 | 326.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 314.95 | 314.81 | 319.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 15:15:00 | 307.40 | 313.92 | 318.29 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 314.00 | 313.86 | 318.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 318.20 | 313.94 | 318.14 | SL hit (close>ema400) qty=1.00 sl=318.14 alert=retest1 |

### Cycle 7 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 332.80 | 320.53 | 320.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 334.65 | 321.08 | 320.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 359.95 | 360.29 | 347.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 359.95 | 360.29 | 347.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 344.60 | 359.32 | 347.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 344.60 | 359.32 | 347.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 344.25 | 359.17 | 347.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 344.30 | 359.17 | 347.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 353.25 | 356.05 | 347.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 358.70 | 355.46 | 347.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 339.25 | 353.74 | 347.37 | SL hit (close<static) qty=1.00 sl=340.05 alert=retest2 |

### Cycle 8 — SELL (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 09:15:00 | 317.60 | 342.83 | 342.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 13:15:00 | 315.00 | 341.80 | 342.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 292.95 | 291.02 | 302.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 09:30:00 | 293.90 | 291.02 | 302.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 299.25 | 291.55 | 302.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:30:00 | 298.95 | 291.55 | 302.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 302.10 | 291.66 | 302.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 302.10 | 291.66 | 302.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 302.40 | 291.76 | 302.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 310.20 | 291.76 | 302.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 311.20 | 291.96 | 302.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 310.00 | 291.96 | 302.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 315.40 | 292.19 | 302.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 312.00 | 292.19 | 302.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 311.50 | 303.42 | 306.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:00:00 | 311.50 | 303.42 | 306.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 310.95 | 304.11 | 306.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:30:00 | 308.75 | 304.11 | 306.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 310.35 | 304.17 | 306.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:30:00 | 307.20 | 304.33 | 306.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 306.20 | 304.33 | 306.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 307.30 | 304.48 | 306.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 291.84 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 290.89 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 291.94 | 304.20 | 306.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 305.25 | 303.90 | 306.38 | SL hit (close>ema200) qty=0.50 sl=303.90 alert=retest2 |

### Cycle 9 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 329.05 | 303.95 | 303.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 12:15:00 | 333.50 | 304.74 | 304.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 316.50 | 316.97 | 312.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 10:00:00 | 316.50 | 316.97 | 312.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 309.30 | 327.90 | 320.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 333.25 | 324.11 | 319.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 335.80 | 322.03 | 320.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 334.65 | 322.03 | 320.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 334.80 | 336.91 | 330.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 331.55 | 336.68 | 330.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:45:00 | 339.10 | 336.67 | 330.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 339.05 | 336.67 | 330.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 323.35 | 336.54 | 330.76 | SL hit (close<static) qty=1.00 sl=328.20 alert=retest2 |

### Cycle 10 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 323.85 | 334.63 | 334.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 322.00 | 334.50 | 334.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 329.95 | 328.98 | 331.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 319.65 | 317.39 | 322.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 320.75 | 317.39 | 322.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 281.05 | 274.04 | 281.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:15:00 | 280.80 | 274.04 | 281.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 280.80 | 274.10 | 281.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 276.95 | 274.10 | 281.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 283.30 | 274.34 | 281.65 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 265.15 | 243.03 | 243.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 266.15 | 243.65 | 243.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-03 10:45:00 | 266.65 | 2023-08-28 12:15:00 | 264.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-08-04 09:15:00 | 268.85 | 2023-08-28 12:15:00 | 264.45 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-08-04 15:00:00 | 266.40 | 2023-08-28 12:15:00 | 264.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-08-07 15:00:00 | 266.70 | 2023-08-28 12:15:00 | 264.45 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-11 10:30:00 | 269.00 | 2023-08-30 12:15:00 | 263.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-08-14 10:00:00 | 267.50 | 2023-08-30 12:15:00 | 263.90 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-08-14 10:30:00 | 267.70 | 2023-09-27 10:15:00 | 293.31 | TARGET_HIT | 1.00 | 9.57% |
| BUY | retest2 | 2023-08-21 14:30:00 | 267.70 | 2023-09-27 10:15:00 | 293.04 | TARGET_HIT | 1.00 | 9.47% |
| BUY | retest2 | 2023-08-22 09:15:00 | 269.85 | 2023-09-27 10:15:00 | 293.37 | TARGET_HIT | 1.00 | 8.72% |
| BUY | retest2 | 2023-08-22 15:00:00 | 268.95 | 2023-09-27 10:15:00 | 294.25 | TARGET_HIT | 1.00 | 9.41% |
| BUY | retest2 | 2023-08-25 09:15:00 | 269.05 | 2023-09-27 10:15:00 | 294.47 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2023-08-28 10:45:00 | 268.50 | 2023-09-27 10:15:00 | 294.47 | TARGET_HIT | 1.00 | 9.67% |
| BUY | retest2 | 2023-08-28 14:15:00 | 266.65 | 2023-09-27 10:15:00 | 293.81 | TARGET_HIT | 1.00 | 10.19% |
| BUY | retest2 | 2023-08-30 09:15:00 | 266.85 | 2023-09-27 10:15:00 | 293.10 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2023-08-31 09:15:00 | 267.10 | 2023-09-28 09:15:00 | 295.74 | TARGET_HIT | 1.00 | 10.72% |
| BUY | retest2 | 2023-08-31 11:15:00 | 266.45 | 2023-09-28 09:15:00 | 295.90 | TARGET_HIT | 1.00 | 11.05% |
| BUY | retest2 | 2023-09-01 11:45:00 | 270.95 | 2023-09-28 09:15:00 | 298.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-04 09:45:00 | 270.85 | 2023-09-28 09:15:00 | 297.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-04 11:45:00 | 270.35 | 2023-09-28 09:15:00 | 297.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-04 12:15:00 | 270.30 | 2023-09-28 09:15:00 | 297.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-06 09:15:00 | 272.95 | 2023-10-11 09:15:00 | 300.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-12 14:15:00 | 272.60 | 2023-10-11 09:15:00 | 299.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-15 09:45:00 | 273.65 | 2023-10-11 09:15:00 | 301.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-15 10:30:00 | 273.35 | 2023-10-11 09:15:00 | 300.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 09:15:00 | 288.55 | 2023-11-30 12:15:00 | 285.95 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-11-02 12:15:00 | 286.10 | 2023-11-30 12:15:00 | 285.95 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2023-11-02 15:15:00 | 289.95 | 2023-11-30 12:15:00 | 285.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-11-07 13:30:00 | 285.70 | 2023-12-08 12:15:00 | 314.71 | TARGET_HIT | 1.00 | 10.15% |
| BUY | retest2 | 2023-11-10 09:15:00 | 297.95 | 2023-12-08 12:15:00 | 314.27 | TARGET_HIT | 1.00 | 5.48% |
| BUY | retest2 | 2023-11-22 09:30:00 | 287.00 | 2023-12-08 12:15:00 | 315.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-22 10:00:00 | 287.70 | 2023-12-08 12:15:00 | 315.65 | TARGET_HIT | 1.00 | 9.71% |
| BUY | retest2 | 2023-11-29 09:15:00 | 286.95 | 2023-12-08 12:15:00 | 315.70 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2023-11-29 12:45:00 | 287.40 | 2023-12-08 13:15:00 | 317.41 | TARGET_HIT | 1.00 | 10.44% |
| BUY | retest2 | 2023-11-29 13:45:00 | 289.20 | 2023-12-08 13:15:00 | 316.47 | TARGET_HIT | 1.00 | 9.43% |
| BUY | retest2 | 2023-11-30 09:15:00 | 289.95 | 2023-12-08 15:15:00 | 318.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-30 14:45:00 | 287.00 | 2023-12-11 09:15:00 | 327.75 | TARGET_HIT | 1.00 | 14.20% |
| BUY | retest2 | 2023-12-01 09:15:00 | 294.40 | 2023-12-11 09:15:00 | 323.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-15 09:15:00 | 316.30 | 2024-04-23 13:15:00 | 332.00 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2024-05-06 09:30:00 | 322.00 | 2024-05-21 09:15:00 | 305.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:15:00 | 321.85 | 2024-05-21 09:15:00 | 305.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:45:00 | 322.35 | 2024-05-21 09:15:00 | 306.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:30:00 | 322.00 | 2024-06-03 11:15:00 | 314.80 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2024-05-06 13:15:00 | 321.85 | 2024-06-03 11:15:00 | 314.80 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2024-05-06 13:45:00 | 322.35 | 2024-06-03 11:15:00 | 314.80 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2024-06-06 12:15:00 | 312.00 | 2024-06-07 13:15:00 | 320.10 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-06-06 13:45:00 | 312.60 | 2024-06-07 13:15:00 | 320.10 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-06-06 14:30:00 | 312.85 | 2024-06-07 13:15:00 | 320.10 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-06-06 15:00:00 | 312.05 | 2024-06-07 13:15:00 | 320.10 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-06-10 12:45:00 | 317.75 | 2024-06-12 10:15:00 | 331.55 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2024-06-10 14:15:00 | 318.50 | 2024-06-12 10:15:00 | 331.55 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-06-11 11:45:00 | 319.00 | 2024-06-12 10:15:00 | 331.55 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest1 | 2024-08-27 15:15:00 | 307.40 | 2024-08-28 14:15:00 | 318.20 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-10-31 15:00:00 | 358.70 | 2024-11-06 12:15:00 | 339.25 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2025-02-12 14:30:00 | 307.20 | 2025-02-17 09:15:00 | 291.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 306.20 | 2025-02-17 09:15:00 | 290.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 307.30 | 2025-02-17 09:15:00 | 291.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 14:30:00 | 307.20 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-02-12 15:15:00 | 306.20 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2025-02-13 15:00:00 | 307.30 | 2025-02-18 14:15:00 | 305.25 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-03-03 09:15:00 | 304.00 | 2025-03-03 15:15:00 | 313.90 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-03-04 09:15:00 | 307.85 | 2025-03-17 11:15:00 | 292.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-04 09:15:00 | 307.85 | 2025-03-18 09:15:00 | 304.70 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2025-03-24 09:30:00 | 306.45 | 2025-03-28 15:15:00 | 292.46 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-03-24 11:00:00 | 307.85 | 2025-04-02 09:15:00 | 291.13 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-03-24 09:30:00 | 306.45 | 2025-04-02 14:15:00 | 301.45 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-03-24 11:00:00 | 307.85 | 2025-04-02 14:15:00 | 301.45 | STOP_HIT | 0.50 | 2.08% |
| BUY | retest2 | 2025-06-10 13:15:00 | 333.25 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-07-08 13:30:00 | 335.80 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-07-08 14:15:00 | 334.65 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-30 09:15:00 | 334.80 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-31 12:45:00 | 339.10 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-07-31 13:15:00 | 339.05 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-08-01 09:30:00 | 340.20 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-08-01 14:45:00 | 338.95 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-08-12 10:45:00 | 337.75 | 2025-09-12 13:15:00 | 332.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-12 11:45:00 | 335.95 | 2025-09-17 11:15:00 | 330.65 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-25 14:30:00 | 336.55 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-08-26 10:00:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-09-05 14:15:00 | 337.00 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-09-15 09:15:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 276.95 | 2026-02-04 11:15:00 | 283.30 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-06 09:15:00 | 261.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-13 09:15:00 | 247.37 | TARGET_HIT | 0.50 | 10.00% |
