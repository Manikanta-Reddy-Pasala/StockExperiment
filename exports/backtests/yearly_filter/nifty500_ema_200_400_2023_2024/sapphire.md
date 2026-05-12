# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 183.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT2_SKIP | 9 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 62 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 53
- **Target hits / Stop hits / Partials:** 4 / 62 / 10
- **Avg / median % per leg:** -0.76% / -2.21%
- **Sum % (uncompounded):** -57.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 3 | 10.0% | 3 | 27 | 0 | -1.63% | -48.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.28% | -9.1% |
| BUY @ 3rd Alert (retest2) | 26 | 3 | 11.5% | 3 | 23 | 0 | -1.53% | -39.7% |
| SELL (all) | 46 | 20 | 43.5% | 1 | 35 | 10 | -0.20% | -9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 20 | 43.5% | 1 | 35 | 10 | -0.20% | -9.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.28% | -9.1% |
| retest2 (combined) | 72 | 23 | 31.9% | 4 | 58 | 10 | -0.68% | -48.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 09:15:00 | 255.02 | 280.68 | 280.70 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 14:15:00 | 285.60 | 276.72 | 276.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 11:15:00 | 286.06 | 277.42 | 277.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 277.00 | 279.97 | 278.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 277.00 | 279.97 | 278.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 277.00 | 279.97 | 278.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 277.00 | 279.97 | 278.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 281.88 | 279.99 | 278.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:00:00 | 283.22 | 280.05 | 278.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 15:15:00 | 285.00 | 280.14 | 278.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:00:00 | 283.54 | 280.76 | 279.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 10:30:00 | 283.72 | 280.98 | 279.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 279.62 | 281.03 | 279.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 279.44 | 281.03 | 279.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 281.59 | 281.04 | 279.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 13:15:00 | 282.58 | 281.04 | 279.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:00:00 | 282.43 | 281.08 | 279.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:45:00 | 282.20 | 281.08 | 279.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:15:00 | 282.12 | 281.08 | 279.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 279.96 | 281.60 | 280.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 279.96 | 281.60 | 280.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 283.09 | 281.62 | 280.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-09 10:15:00 | 277.81 | 281.55 | 280.09 | SL hit (close<static) qty=1.00 sl=279.49 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 12:15:00 | 270.40 | 281.43 | 281.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 13:15:00 | 266.61 | 281.28 | 281.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 280.81 | 279.27 | 280.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 280.81 | 279.27 | 280.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 280.81 | 279.27 | 280.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 280.81 | 279.27 | 280.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 280.54 | 279.28 | 280.32 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 13:15:00 | 303.91 | 281.42 | 281.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 10:15:00 | 305.00 | 290.34 | 286.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 11:15:00 | 289.60 | 291.46 | 287.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-12 11:45:00 | 289.58 | 291.46 | 287.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 287.93 | 291.39 | 287.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 15:00:00 | 287.93 | 291.39 | 287.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 287.66 | 291.35 | 287.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 290.33 | 291.35 | 287.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-13 11:15:00 | 280.31 | 291.14 | 287.33 | SL hit (close<static) qty=1.00 sl=285.04 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 10:15:00 | 281.81 | 293.34 | 293.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 279.90 | 292.63 | 292.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 15:15:00 | 285.00 | 284.12 | 287.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:15:00 | 282.70 | 284.12 | 287.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 288.00 | 283.49 | 286.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 288.00 | 283.49 | 286.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 285.69 | 283.51 | 286.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:30:00 | 284.64 | 283.52 | 286.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 284.76 | 283.54 | 286.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:00:00 | 285.03 | 283.55 | 286.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:30:00 | 283.20 | 283.55 | 286.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 292.00 | 283.58 | 286.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 292.00 | 283.58 | 286.55 | SL hit (close>static) qty=1.00 sl=288.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 301.87 | 288.46 | 288.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 15:15:00 | 303.00 | 290.58 | 289.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 305.60 | 306.50 | 300.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 12:00:00 | 308.06 | 306.51 | 300.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:45:00 | 308.06 | 307.59 | 301.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:15:00 | 308.60 | 307.59 | 301.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 12:30:00 | 307.97 | 307.70 | 301.80 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 301.15 | 307.52 | 302.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 301.15 | 307.52 | 302.03 | SL hit (close<ema400) qty=1.00 sl=302.03 alert=retest1 |

### Cycle 7 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 317.55 | 334.03 | 334.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 315.90 | 333.85 | 333.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 321.65 | 320.39 | 325.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 321.65 | 320.39 | 325.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 321.65 | 320.39 | 325.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:30:00 | 314.45 | 327.10 | 328.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:30:00 | 315.65 | 325.59 | 327.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 314.60 | 325.43 | 327.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 326.15 | 321.74 | 324.86 | SL hit (close>static) qty=1.00 sl=326.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 345.65 | 327.01 | 326.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 365.50 | 327.81 | 327.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 329.15 | 332.56 | 330.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 13:15:00 | 329.15 | 332.56 | 330.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 329.15 | 332.56 | 330.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 329.15 | 332.56 | 330.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 328.45 | 332.52 | 330.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 328.45 | 332.52 | 330.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 331.95 | 332.52 | 330.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 318.90 | 332.52 | 330.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 320.35 | 332.40 | 330.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 15:15:00 | 331.50 | 331.20 | 329.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 11:15:00 | 328.90 | 331.04 | 329.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 330.95 | 331.00 | 329.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 313.10 | 329.46 | 328.81 | SL hit (close<static) qty=1.00 sl=315.20 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 313.85 | 328.17 | 328.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 310.35 | 326.90 | 327.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 316.20 | 313.61 | 319.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 14:00:00 | 316.20 | 313.61 | 319.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 315.50 | 313.63 | 319.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:45:00 | 319.65 | 313.63 | 319.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 322.60 | 313.70 | 319.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 322.60 | 313.70 | 319.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 326.80 | 313.83 | 319.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:15:00 | 334.00 | 313.83 | 319.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 329.45 | 314.34 | 320.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 15:00:00 | 325.15 | 314.44 | 320.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 09:15:00 | 308.89 | 315.13 | 319.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 315.45 | 315.06 | 319.71 | SL hit (close>ema200) qty=0.50 sl=315.06 alert=retest2 |

### Cycle 10 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 321.10 | 311.20 | 311.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 323.80 | 311.57 | 311.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 309.10 | 313.04 | 312.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 309.10 | 313.04 | 312.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 309.10 | 313.04 | 312.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 315.50 | 312.98 | 312.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 305.80 | 312.91 | 312.13 | SL hit (close<static) qty=1.00 sl=306.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 10:15:00 | 308.00 | 311.38 | 311.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 11:15:00 | 303.50 | 311.03 | 311.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 310.55 | 310.21 | 310.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 321.05 | 310.32 | 310.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 321.05 | 310.32 | 310.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 325.15 | 310.46 | 310.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:45:00 | 325.15 | 310.46 | 310.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 335.65 | 311.50 | 311.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 349.40 | 323.00 | 319.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 326.60 | 327.19 | 322.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:45:00 | 327.20 | 327.19 | 322.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 330.90 | 331.17 | 326.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 329.90 | 331.17 | 326.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 326.60 | 331.88 | 326.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 326.60 | 331.88 | 326.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 328.40 | 331.84 | 326.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 331.20 | 331.84 | 326.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 324.15 | 331.44 | 327.06 | SL hit (close<static) qty=1.00 sl=325.25 alert=retest2 |

### Cycle 13 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 318.05 | 323.86 | 323.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 310.00 | 323.25 | 323.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 323.45 | 321.42 | 322.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 325.95 | 321.46 | 322.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 325.95 | 321.46 | 322.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 326.55 | 321.51 | 322.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 322.05 | 321.51 | 322.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:30:00 | 324.00 | 321.66 | 322.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 331.70 | 321.92 | 322.76 | SL hit (close>static) qty=1.00 sl=328.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 325.75 | 323.41 | 323.40 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 321.65 | 323.39 | 323.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 321.10 | 323.37 | 323.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 335.00 | 323.00 | 323.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 333.10 | 323.39 | 323.38 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 311.90 | 323.52 | 323.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 309.05 | 323.00 | 323.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 290.60 | 295.09 | 303.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 276.07 | 292.27 | 301.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-11 09:15:00 | 261.54 | 288.97 | 298.93 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-21 10:00:00 | 283.22 | 2024-01-09 10:15:00 | 277.81 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-12-22 15:15:00 | 285.00 | 2024-01-09 10:15:00 | 277.81 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2023-12-29 11:00:00 | 283.54 | 2024-01-09 10:15:00 | 277.81 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-01-01 10:30:00 | 283.72 | 2024-01-09 10:15:00 | 277.81 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-01-02 13:15:00 | 282.58 | 2024-01-09 15:15:00 | 276.25 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-01-03 10:00:00 | 282.43 | 2024-01-09 15:15:00 | 276.25 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-01-03 13:45:00 | 282.20 | 2024-01-09 15:15:00 | 276.25 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-01-03 14:15:00 | 282.12 | 2024-01-09 15:15:00 | 276.25 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-01-10 10:45:00 | 285.00 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-01-10 12:15:00 | 284.98 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-01-10 13:00:00 | 286.16 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-01-12 13:00:00 | 287.76 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-01-24 10:15:00 | 289.96 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2024-01-30 13:15:00 | 288.56 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-01-31 09:15:00 | 291.72 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2024-02-01 13:00:00 | 290.00 | 2024-02-05 10:15:00 | 278.43 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-02-06 13:30:00 | 283.55 | 2024-02-07 09:15:00 | 279.16 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-03-13 09:15:00 | 290.33 | 2024-03-13 11:15:00 | 280.31 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-03-18 09:15:00 | 294.49 | 2024-03-22 09:15:00 | 323.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-29 11:30:00 | 284.64 | 2024-05-31 14:15:00 | 292.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-05-29 13:15:00 | 284.76 | 2024-05-31 14:15:00 | 292.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-05-29 14:00:00 | 285.03 | 2024-05-31 14:15:00 | 292.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-05-29 14:30:00 | 283.20 | 2024-05-31 14:15:00 | 292.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-06-03 10:45:00 | 286.00 | 2024-06-05 12:15:00 | 292.18 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-06-03 12:15:00 | 286.87 | 2024-06-05 12:15:00 | 292.18 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-06-04 09:30:00 | 286.79 | 2024-06-05 13:15:00 | 295.80 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-06-04 11:00:00 | 281.71 | 2024-06-05 13:15:00 | 295.80 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2024-06-04 15:15:00 | 281.20 | 2024-06-05 13:15:00 | 295.80 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2024-06-05 09:30:00 | 276.98 | 2024-06-05 13:15:00 | 295.80 | STOP_HIT | 1.00 | -6.79% |
| BUY | retest1 | 2024-07-10 12:00:00 | 308.06 | 2024-07-22 09:15:00 | 301.15 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest1 | 2024-07-16 09:45:00 | 308.06 | 2024-07-22 09:15:00 | 301.15 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest1 | 2024-07-16 11:15:00 | 308.60 | 2024-07-22 09:15:00 | 301.15 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2024-07-18 12:30:00 | 307.97 | 2024-07-22 09:15:00 | 301.15 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-07-22 10:30:00 | 302.28 | 2024-07-31 14:15:00 | 332.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:45:00 | 303.35 | 2024-07-31 14:15:00 | 333.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-13 09:30:00 | 314.45 | 2024-12-24 09:15:00 | 326.15 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2024-12-16 13:30:00 | 315.65 | 2024-12-24 09:15:00 | 326.15 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-12-17 09:15:00 | 314.60 | 2024-12-24 09:15:00 | 326.15 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-01-14 15:15:00 | 331.50 | 2025-01-20 09:15:00 | 313.10 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest2 | 2025-01-15 11:15:00 | 328.90 | 2025-01-20 09:15:00 | 313.10 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2025-01-15 13:00:00 | 330.95 | 2025-01-20 09:15:00 | 313.10 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2025-02-03 15:00:00 | 325.15 | 2025-02-07 09:15:00 | 308.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 15:00:00 | 325.15 | 2025-02-07 13:15:00 | 315.45 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-02-21 10:00:00 | 324.65 | 2025-02-27 14:15:00 | 308.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 11:00:00 | 326.45 | 2025-02-27 14:15:00 | 310.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 12:30:00 | 326.00 | 2025-02-27 14:15:00 | 309.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 317.85 | 2025-02-28 11:15:00 | 301.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 12:30:00 | 317.25 | 2025-02-28 11:15:00 | 301.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:30:00 | 317.90 | 2025-02-28 11:15:00 | 302.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:45:00 | 317.60 | 2025-02-28 11:15:00 | 301.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:00:00 | 324.65 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-02-21 11:00:00 | 326.45 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-02-21 12:30:00 | 326.00 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-02-25 11:15:00 | 317.85 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2025-02-25 12:30:00 | 317.25 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 0.24% |
| SELL | retest2 | 2025-02-27 09:30:00 | 317.90 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2025-02-27 11:45:00 | 317.60 | 2025-03-03 12:15:00 | 316.50 | STOP_HIT | 0.50 | 0.35% |
| SELL | retest2 | 2025-03-05 12:00:00 | 315.70 | 2025-03-06 09:15:00 | 339.75 | STOP_HIT | 1.00 | -7.62% |
| SELL | retest2 | 2025-03-11 09:15:00 | 307.80 | 2025-03-11 14:15:00 | 323.55 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2025-03-12 11:15:00 | 315.45 | 2025-03-17 10:15:00 | 299.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 11:15:00 | 315.45 | 2025-03-21 14:15:00 | 313.05 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2025-04-21 13:15:00 | 315.75 | 2025-04-22 14:15:00 | 321.40 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-07 14:30:00 | 315.50 | 2025-05-07 15:15:00 | 305.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-07-28 11:45:00 | 331.20 | 2025-07-30 13:15:00 | 324.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-20 09:15:00 | 322.05 | 2025-08-21 10:15:00 | 331.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-08-20 12:30:00 | 324.00 | 2025-08-21 10:15:00 | 331.70 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-26 13:45:00 | 324.50 | 2025-09-01 12:15:00 | 326.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-29 10:45:00 | 324.55 | 2025-09-01 12:15:00 | 326.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-09-01 10:15:00 | 323.45 | 2025-09-02 09:15:00 | 325.65 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-01 11:00:00 | 323.20 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-09-02 09:15:00 | 323.50 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-09-02 10:15:00 | 323.30 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-09-02 11:30:00 | 322.55 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-09-03 13:45:00 | 322.55 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-10-31 11:00:00 | 290.60 | 2025-11-07 09:15:00 | 276.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:00:00 | 290.60 | 2025-11-11 09:15:00 | 261.54 | TARGET_HIT | 0.50 | 10.00% |
