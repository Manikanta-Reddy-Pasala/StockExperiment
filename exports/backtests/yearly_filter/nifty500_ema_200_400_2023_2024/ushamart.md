# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 74 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 50
- **Target hits / Stop hits / Partials:** 5 / 54 / 11
- **Avg / median % per leg:** -0.39% / -1.92%
- **Sum % (uncompounded):** -26.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 4 | 10 | 0 | 0.94% | 13.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 4 | 10 | 0 | 0.94% | 13.2% |
| SELL (all) | 56 | 16 | 28.6% | 1 | 44 | 11 | -0.72% | -40.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 56 | 16 | 28.6% | 1 | 44 | 11 | -0.72% | -40.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 20 | 28.6% | 5 | 54 | 11 | -0.39% | -27.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 11:15:00 | 268.30 | 324.64 | 324.86 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 333.40 | 319.60 | 319.55 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 305.65 | 319.72 | 319.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 299.95 | 319.53 | 319.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 316.75 | 314.03 | 316.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 316.75 | 314.03 | 316.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 316.75 | 314.03 | 316.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 316.75 | 314.03 | 316.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 313.30 | 314.03 | 316.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 12:15:00 | 312.40 | 314.03 | 316.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-21 09:15:00 | 296.78 | 313.60 | 316.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 313.60 | 312.70 | 315.60 | SL hit (close>ema200) qty=0.50 sl=312.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 10:15:00 | 364.30 | 313.96 | 313.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 372.40 | 325.08 | 319.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 331.80 | 336.95 | 327.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 10:15:00 | 331.80 | 336.95 | 327.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 331.80 | 336.95 | 327.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 331.80 | 336.95 | 327.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 328.00 | 336.80 | 327.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:00:00 | 328.00 | 336.80 | 327.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 332.00 | 336.75 | 327.94 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 307.40 | 323.51 | 323.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 304.30 | 320.84 | 322.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 307.40 | 306.09 | 313.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:00:00 | 307.40 | 306.09 | 313.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 312.35 | 306.25 | 313.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:30:00 | 312.00 | 306.25 | 313.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 313.20 | 306.36 | 313.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:30:00 | 315.70 | 306.36 | 313.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 311.50 | 306.42 | 313.07 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 14:15:00 | 342.90 | 318.09 | 317.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 12:15:00 | 346.20 | 321.93 | 320.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 344.95 | 349.06 | 338.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 10:00:00 | 344.95 | 349.06 | 338.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 339.00 | 348.82 | 338.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:45:00 | 340.10 | 348.82 | 338.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 340.90 | 348.75 | 338.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 13:45:00 | 343.15 | 348.03 | 338.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 333.50 | 347.76 | 338.08 | SL hit (close<static) qty=1.00 sl=337.15 alert=retest2 |

### Cycle 7 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 335.95 | 368.17 | 368.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 331.00 | 365.87 | 367.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 350.70 | 345.86 | 353.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:00:00 | 350.70 | 345.86 | 353.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 354.35 | 345.99 | 353.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 351.70 | 345.99 | 353.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 360.00 | 346.13 | 353.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 360.00 | 346.13 | 353.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 360.10 | 346.27 | 353.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:30:00 | 362.80 | 346.27 | 353.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 353.60 | 349.51 | 354.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 354.20 | 349.51 | 354.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 356.70 | 349.58 | 354.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 356.70 | 349.58 | 354.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 357.45 | 349.66 | 354.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:30:00 | 357.80 | 349.66 | 354.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 359.95 | 350.59 | 354.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 359.95 | 350.59 | 354.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 359.65 | 350.68 | 354.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 359.25 | 350.68 | 354.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 356.05 | 351.03 | 354.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 351.85 | 351.37 | 354.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:45:00 | 353.15 | 350.49 | 354.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:45:00 | 352.30 | 350.51 | 354.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 14:30:00 | 352.55 | 350.40 | 353.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 352.50 | 350.42 | 353.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 354.20 | 350.42 | 353.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 352.10 | 350.44 | 353.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 353.15 | 350.44 | 353.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 352.95 | 350.45 | 353.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:00:00 | 352.95 | 350.45 | 353.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 351.45 | 350.48 | 353.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 359.05 | 350.89 | 353.93 | SL hit (close>static) qty=1.00 sl=359.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 422.45 | 355.48 | 355.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 10:15:00 | 428.50 | 357.48 | 356.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 10:15:00 | 383.90 | 384.12 | 372.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 10:45:00 | 383.65 | 384.12 | 372.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 386.85 | 398.43 | 385.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 386.85 | 398.43 | 385.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 383.50 | 398.28 | 385.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 383.50 | 398.28 | 385.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 383.00 | 398.13 | 385.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 376.40 | 398.13 | 385.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 369.45 | 397.84 | 385.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 369.55 | 397.84 | 385.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 384.75 | 395.78 | 384.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 378.60 | 395.78 | 384.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 385.10 | 395.67 | 384.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:45:00 | 381.65 | 395.67 | 384.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 381.35 | 395.53 | 384.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:00:00 | 381.35 | 395.53 | 384.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 380.60 | 395.38 | 384.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 379.45 | 395.38 | 384.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 383.30 | 394.40 | 384.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 383.95 | 394.40 | 384.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 384.00 | 394.30 | 384.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 397.60 | 394.30 | 384.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 385.85 | 394.14 | 385.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:00:00 | 385.65 | 394.05 | 385.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 378.45 | 393.68 | 385.56 | SL hit (close<static) qty=1.00 sl=381.55 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 376.85 | 385.85 | 385.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 10:15:00 | 374.05 | 385.73 | 385.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 315.00 | 314.35 | 333.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:30:00 | 315.55 | 314.35 | 333.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 332.35 | 315.00 | 332.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:45:00 | 332.20 | 315.00 | 332.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 333.55 | 315.19 | 332.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:00:00 | 333.55 | 315.19 | 332.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 334.45 | 315.38 | 332.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:45:00 | 334.15 | 315.38 | 332.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 329.60 | 315.82 | 332.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 12:15:00 | 327.80 | 317.17 | 332.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:15:00 | 311.41 | 317.60 | 331.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 323.00 | 316.55 | 330.00 | SL hit (close>ema200) qty=0.50 sl=316.55 alert=retest2 |

### Cycle 10 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 346.30 | 316.49 | 316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 351.00 | 316.84 | 316.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 360.60 | 362.75 | 347.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 360.60 | 362.75 | 347.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 352.10 | 366.00 | 352.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 352.10 | 366.00 | 352.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 349.20 | 365.83 | 352.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 350.75 | 365.83 | 352.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 347.20 | 364.70 | 352.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 347.20 | 364.70 | 352.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 351.30 | 361.04 | 351.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 352.60 | 359.85 | 351.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 344.45 | 359.49 | 351.66 | SL hit (close<static) qty=1.00 sl=346.60 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 421.40 | 438.71 | 438.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 414.30 | 437.88 | 438.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 431.95 | 423.88 | 429.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 433.35 | 423.98 | 429.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 432.70 | 423.98 | 429.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 427.70 | 425.03 | 430.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 428.20 | 425.03 | 430.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 423.70 | 425.02 | 430.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 420.50 | 424.96 | 429.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 416.65 | 424.91 | 429.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 422.00 | 424.00 | 428.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 420.50 | 423.99 | 428.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.30 | 423.74 | 428.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 426.30 | 423.74 | 428.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 400.90 | 421.23 | 426.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 399.47 | 419.44 | 425.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 399.47 | 419.44 | 425.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | SL hit (close>ema200) qty=0.50 sl=419.19 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 448.80 | 419.91 | 419.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 450.55 | 421.35 | 420.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-12-20 12:15:00 | 312.40 | 2023-12-21 09:15:00 | 296.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 12:15:00 | 312.40 | 2023-12-26 09:15:00 | 313.60 | STOP_HIT | 0.50 | -0.38% |
| SELL | retest2 | 2023-12-26 10:15:00 | 311.95 | 2023-12-28 14:15:00 | 296.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-26 11:30:00 | 310.80 | 2023-12-29 10:15:00 | 295.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-26 12:15:00 | 309.80 | 2023-12-29 10:15:00 | 294.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-26 10:15:00 | 311.95 | 2024-01-04 12:15:00 | 313.80 | STOP_HIT | 0.50 | -0.59% |
| SELL | retest2 | 2023-12-26 11:30:00 | 310.80 | 2024-01-04 12:15:00 | 313.80 | STOP_HIT | 0.50 | -0.97% |
| SELL | retest2 | 2023-12-26 12:15:00 | 309.80 | 2024-01-04 12:15:00 | 313.80 | STOP_HIT | 0.50 | -1.29% |
| SELL | retest2 | 2024-01-04 14:30:00 | 311.60 | 2024-01-05 12:15:00 | 314.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-01-05 10:15:00 | 312.50 | 2024-01-05 12:15:00 | 314.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-01-05 14:00:00 | 310.85 | 2024-01-09 10:15:00 | 295.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-05 14:00:00 | 310.85 | 2024-01-10 13:15:00 | 307.90 | STOP_HIT | 0.50 | 0.95% |
| SELL | retest2 | 2024-01-15 12:45:00 | 311.45 | 2024-01-16 09:15:00 | 313.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-01-15 15:00:00 | 308.75 | 2024-01-16 10:15:00 | 316.35 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-01-16 13:15:00 | 310.65 | 2024-01-16 14:15:00 | 317.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-01-17 12:00:00 | 310.75 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-01-17 12:30:00 | 310.50 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-01-18 09:15:00 | 307.80 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-01-18 13:00:00 | 307.45 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-01-18 13:30:00 | 307.85 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-01-18 14:15:00 | 307.25 | 2024-01-19 10:15:00 | 320.05 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2024-05-10 13:45:00 | 343.15 | 2024-05-13 09:15:00 | 333.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-05-14 11:00:00 | 343.70 | 2024-05-15 14:15:00 | 336.40 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-05-14 12:30:00 | 344.50 | 2024-05-15 14:15:00 | 336.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-05-17 12:15:00 | 342.95 | 2024-06-03 09:15:00 | 377.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 12:45:00 | 341.85 | 2024-06-11 09:15:00 | 376.04 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-18 09:15:00 | 351.85 | 2024-09-26 09:15:00 | 359.05 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-09-20 09:45:00 | 353.15 | 2024-09-26 09:15:00 | 359.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-09-20 10:45:00 | 352.30 | 2024-09-26 09:15:00 | 359.05 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-23 14:30:00 | 352.55 | 2024-09-26 09:15:00 | 359.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-09-30 11:45:00 | 349.45 | 2024-09-30 12:15:00 | 355.20 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-10-04 14:15:00 | 349.60 | 2024-10-10 10:15:00 | 355.15 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-07 09:30:00 | 348.60 | 2024-10-10 10:15:00 | 355.15 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-10-07 10:30:00 | 343.75 | 2024-10-10 10:15:00 | 355.15 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-11-19 09:15:00 | 397.60 | 2024-11-26 13:15:00 | 378.45 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2024-11-26 09:15:00 | 385.85 | 2024-11-26 13:15:00 | 378.45 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-11-26 10:00:00 | 385.65 | 2024-11-26 13:15:00 | 378.45 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-11-27 14:30:00 | 389.00 | 2024-11-28 13:15:00 | 381.35 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-12-03 14:45:00 | 392.80 | 2024-12-18 09:15:00 | 379.65 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-12-16 11:15:00 | 392.50 | 2024-12-18 09:15:00 | 379.65 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-03-10 12:15:00 | 327.80 | 2025-03-13 10:15:00 | 311.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 12:15:00 | 327.80 | 2025-03-18 11:15:00 | 323.00 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2025-03-20 10:30:00 | 327.75 | 2025-03-21 10:15:00 | 335.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-03-20 12:30:00 | 328.30 | 2025-03-21 10:15:00 | 335.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-03-20 13:15:00 | 328.45 | 2025-03-21 10:15:00 | 335.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-04-03 15:00:00 | 330.30 | 2025-04-07 09:15:00 | 297.27 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-19 13:30:00 | 328.80 | 2025-05-20 11:15:00 | 338.15 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-05-20 09:45:00 | 330.40 | 2025-05-20 11:15:00 | 338.15 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-05-21 09:15:00 | 327.15 | 2025-05-28 11:15:00 | 310.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 09:15:00 | 327.15 | 2025-06-09 09:15:00 | 313.20 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-05-28 11:30:00 | 311.70 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -6.66% |
| SELL | retest2 | 2025-06-09 15:00:00 | 315.55 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-06-10 09:15:00 | 314.30 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2025-06-23 09:15:00 | 311.80 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -6.62% |
| SELL | retest2 | 2025-06-23 10:30:00 | 309.50 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -7.42% |
| BUY | retest2 | 2025-08-14 10:15:00 | 352.60 | 2025-08-14 12:15:00 | 344.45 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-14 14:45:00 | 354.55 | 2025-08-25 13:15:00 | 387.48 | TARGET_HIT | 1.00 | 9.29% |
| BUY | retest2 | 2025-08-18 11:15:00 | 352.25 | 2025-08-26 09:15:00 | 390.01 | TARGET_HIT | 1.00 | 10.72% |
| SELL | retest2 | 2026-02-11 13:00:00 | 420.50 | 2026-02-20 15:15:00 | 400.90 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-02-13 09:15:00 | 416.65 | 2026-02-24 12:15:00 | 399.47 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-02-16 11:15:00 | 422.00 | 2026-02-24 12:15:00 | 399.47 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2026-02-11 13:00:00 | 420.50 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -3.63% |
| SELL | retest2 | 2026-02-13 09:15:00 | 416.65 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -4.58% |
| SELL | retest2 | 2026-02-16 11:15:00 | 422.00 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -3.26% |
| SELL | retest2 | 2026-02-16 12:15:00 | 420.50 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-02-25 11:15:00 | 429.45 | 2026-03-02 09:15:00 | 407.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 429.45 | 2026-03-02 10:15:00 | 421.65 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2026-04-09 12:45:00 | 433.20 | 2026-04-10 14:15:00 | 442.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-10 10:30:00 | 433.85 | 2026-04-10 14:15:00 | 442.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-04-13 10:00:00 | 433.00 | 2026-04-15 12:15:00 | 441.20 | STOP_HIT | 1.00 | -1.89% |
