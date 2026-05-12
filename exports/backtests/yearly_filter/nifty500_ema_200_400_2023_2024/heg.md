# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT2_SKIP | 6 |
| ALERT3 | 88 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 11 |
| TARGET_HIT | 17 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 47
- **Target hits / Stop hits / Partials:** 17 / 49 / 11
- **Avg / median % per leg:** 1.73% / -0.80%
- **Sum % (uncompounded):** 133.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 10 | 20.0% | 10 | 40 | 0 | 0.55% | 27.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 50 | 10 | 20.0% | 10 | 40 | 0 | 0.55% | 27.7% |
| SELL (all) | 27 | 20 | 74.1% | 7 | 9 | 11 | 3.92% | 105.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 20 | 74.1% | 7 | 9 | 11 | 3.92% | 105.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 77 | 30 | 39.0% | 17 | 49 | 11 | 1.73% | 133.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 323.50 | 336.83 | 336.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 316.00 | 335.56 | 336.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 326.80 | 325.43 | 330.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-29 10:00:00 | 326.80 | 325.43 | 330.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 329.45 | 325.45 | 329.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 12:00:00 | 329.45 | 325.45 | 329.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 326.70 | 325.46 | 329.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 13:30:00 | 325.84 | 325.45 | 329.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-01 09:15:00 | 333.00 | 325.72 | 329.83 | SL hit (close>static) qty=1.00 sl=331.80 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 13:15:00 | 349.48 | 333.11 | 333.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 358.47 | 336.04 | 334.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 358.81 | 360.29 | 350.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-10 11:00:00 | 358.81 | 360.29 | 350.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 358.80 | 363.22 | 354.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:00:00 | 358.80 | 363.22 | 354.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 349.39 | 362.78 | 354.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:30:00 | 347.09 | 362.78 | 354.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 349.01 | 362.64 | 354.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 351.56 | 362.64 | 354.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 350.33 | 362.40 | 354.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:45:00 | 349.78 | 362.40 | 354.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 344.90 | 361.52 | 354.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 344.90 | 361.52 | 354.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 351.00 | 358.20 | 353.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:45:00 | 350.71 | 358.20 | 353.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 349.38 | 358.04 | 353.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:00:00 | 349.38 | 358.04 | 353.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 357.44 | 365.55 | 358.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 357.44 | 365.55 | 358.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 361.44 | 365.51 | 358.84 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 325.81 | 353.46 | 353.54 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 15:15:00 | 359.40 | 350.31 | 350.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 383.20 | 350.64 | 350.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 14:15:00 | 445.79 | 472.00 | 441.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 14:15:00 | 445.79 | 472.00 | 441.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 445.79 | 472.00 | 441.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 445.79 | 472.00 | 441.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 454.00 | 471.63 | 441.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 443.66 | 471.63 | 441.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 443.60 | 465.82 | 443.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 443.60 | 465.82 | 443.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 444.27 | 465.61 | 443.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:30:00 | 447.88 | 465.26 | 443.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 426.44 | 463.97 | 444.01 | SL hit (close<static) qty=1.00 sl=440.88 alert=retest2 |

### Cycle 5 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 415.04 | 440.12 | 440.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 407.21 | 439.09 | 439.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 10:15:00 | 453.91 | 431.52 | 435.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 10:15:00 | 453.91 | 431.52 | 435.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 453.91 | 431.52 | 435.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 453.91 | 431.52 | 435.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 444.58 | 431.65 | 435.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 443.10 | 431.92 | 435.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:30:00 | 442.44 | 431.99 | 435.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:00:00 | 442.02 | 433.70 | 436.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 440.64 | 433.90 | 436.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 437.11 | 433.96 | 436.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 439.32 | 433.96 | 436.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 436.18 | 433.98 | 436.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 13:45:00 | 435.18 | 434.00 | 436.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 420.94 | 433.80 | 435.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 420.32 | 433.80 | 435.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 419.92 | 433.80 | 435.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 418.61 | 433.80 | 435.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 413.42 | 433.80 | 435.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-06 14:15:00 | 398.79 | 431.46 | 434.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 464.28 | 422.24 | 422.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 499.30 | 423.77 | 422.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 12:15:00 | 443.48 | 447.20 | 436.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 13:00:00 | 443.48 | 447.20 | 436.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 446.00 | 469.60 | 453.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 446.00 | 469.60 | 453.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 445.10 | 469.36 | 452.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 437.90 | 469.36 | 452.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 445.15 | 467.25 | 452.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 445.15 | 467.25 | 452.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 446.05 | 467.04 | 452.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 12:30:00 | 447.15 | 466.84 | 452.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 13:30:00 | 447.45 | 466.62 | 452.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 15:15:00 | 438.30 | 466.12 | 452.24 | SL hit (close<static) qty=1.00 sl=444.25 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 405.65 | 443.86 | 443.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 405.10 | 443.48 | 443.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 11:15:00 | 431.50 | 430.36 | 436.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:45:00 | 431.45 | 430.36 | 436.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 440.00 | 430.49 | 436.00 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 563.30 | 440.71 | 440.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 576.10 | 443.23 | 441.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 13:15:00 | 514.80 | 516.90 | 489.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 13:30:00 | 515.70 | 516.90 | 489.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 493.90 | 517.05 | 495.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 491.90 | 517.05 | 495.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 487.15 | 516.75 | 495.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 487.15 | 516.75 | 495.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 489.00 | 516.48 | 495.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 498.50 | 516.48 | 495.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 491.00 | 515.98 | 495.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 13:00:00 | 494.60 | 515.51 | 495.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 14:00:00 | 496.05 | 515.31 | 495.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 10:15:00 | 485.00 | 514.28 | 495.25 | SL hit (close<static) qty=1.00 sl=485.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 12:15:00 | 431.95 | 481.39 | 481.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 427.60 | 479.39 | 480.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 388.05 | 373.37 | 402.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-06 09:45:00 | 388.45 | 373.37 | 402.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 404.60 | 376.71 | 401.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 412.50 | 376.71 | 401.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 414.40 | 377.08 | 401.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 414.40 | 377.08 | 401.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 411.55 | 378.78 | 401.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 411.50 | 378.78 | 401.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 409.45 | 379.08 | 401.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 413.85 | 379.08 | 401.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 398.90 | 380.84 | 401.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:30:00 | 396.00 | 382.08 | 401.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 396.65 | 382.08 | 401.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 405.45 | 382.67 | 401.83 | SL hit (close>static) qty=1.00 sl=405.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 477.00 | 414.67 | 414.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 483.95 | 417.19 | 415.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 15:15:00 | 458.50 | 458.91 | 444.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 460.55 | 458.91 | 444.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 442.25 | 458.88 | 445.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 441.35 | 458.88 | 445.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 435.55 | 458.65 | 445.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:45:00 | 438.70 | 458.65 | 445.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 444.65 | 458.23 | 445.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 444.65 | 458.23 | 445.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 446.00 | 458.11 | 445.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:30:00 | 443.45 | 458.11 | 445.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 446.25 | 457.99 | 445.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:15:00 | 447.50 | 457.99 | 445.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 440.70 | 457.43 | 445.53 | SL hit (close<static) qty=1.00 sl=444.20 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 461.20 | 505.40 | 505.44 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 519.20 | 504.12 | 504.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 526.00 | 505.15 | 504.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 506.70 | 510.39 | 507.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 506.20 | 510.35 | 507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 506.20 | 510.35 | 507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 507.00 | 510.32 | 507.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 514.25 | 510.32 | 507.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 12:15:00 | 509.50 | 510.34 | 507.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 508.00 | 510.30 | 507.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 509.15 | 510.28 | 507.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 506.25 | 510.24 | 507.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 506.05 | 510.24 | 507.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 505.10 | 510.19 | 507.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 505.10 | 510.19 | 507.69 | SL hit (close<static) qty=1.00 sl=505.15 alert=retest2 |

### Cycle 13 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 524.80 | 549.78 | 549.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 523.65 | 549.52 | 549.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 552.40 | 547.16 | 548.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 551.20 | 547.20 | 548.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 551.20 | 547.20 | 548.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 558.20 | 547.31 | 548.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:00:00 | 558.20 | 547.31 | 548.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 548.85 | 547.62 | 548.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:00:00 | 545.50 | 547.68 | 548.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 553.90 | 547.70 | 548.70 | SL hit (close>static) qty=1.00 sl=552.50 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 569.00 | 549.66 | 549.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 576.25 | 549.92 | 549.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 10:15:00 | 516.60 | 550.41 | 550.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 504.40 | 545.45 | 547.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 560.60 | 522.48 | 533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 570.70 | 522.96 | 534.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 571.95 | 522.96 | 534.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 590.70 | 541.66 | 541.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 607.40 | 544.19 | 542.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 588.55 | 599.54 | 575.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:30:00 | 596.45 | 599.54 | 575.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-29 13:30:00 | 325.84 | 2023-12-01 09:15:00 | 333.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-05-31 13:30:00 | 447.88 | 2024-06-04 10:15:00 | 426.44 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2024-06-11 10:45:00 | 446.98 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-06-11 13:00:00 | 447.30 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-12 09:15:00 | 448.95 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-06-13 13:00:00 | 443.20 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-13 14:00:00 | 443.18 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-14 09:15:00 | 444.40 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-06-19 12:00:00 | 444.69 | 2024-06-25 14:15:00 | 440.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-06-26 09:15:00 | 441.57 | 2024-06-27 12:15:00 | 435.17 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-06-26 10:15:00 | 441.00 | 2024-06-27 12:15:00 | 435.17 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-26 13:45:00 | 442.52 | 2024-06-27 12:15:00 | 435.17 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-26 15:00:00 | 441.55 | 2024-06-27 12:15:00 | 435.17 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-07-02 09:15:00 | 445.94 | 2024-07-03 09:15:00 | 441.38 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-03 09:15:00 | 444.70 | 2024-07-03 09:15:00 | 441.38 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-07-03 10:30:00 | 445.25 | 2024-07-09 13:15:00 | 440.70 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-04 09:15:00 | 446.46 | 2024-07-09 13:15:00 | 440.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-07-30 14:00:00 | 443.10 | 2024-08-05 09:15:00 | 420.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 14:30:00 | 442.44 | 2024-08-05 09:15:00 | 420.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 14:00:00 | 442.02 | 2024-08-05 09:15:00 | 419.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 09:15:00 | 440.64 | 2024-08-05 09:15:00 | 418.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 13:45:00 | 435.18 | 2024-08-05 09:15:00 | 413.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 14:00:00 | 443.10 | 2024-08-06 14:15:00 | 398.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-30 14:30:00 | 442.44 | 2024-08-06 14:15:00 | 398.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-01 14:00:00 | 442.02 | 2024-08-06 14:15:00 | 397.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-02 09:15:00 | 440.64 | 2024-08-06 14:15:00 | 396.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-02 13:45:00 | 435.18 | 2024-08-13 09:15:00 | 432.07 | STOP_HIT | 0.50 | 0.71% |
| SELL | retest2 | 2024-08-13 12:00:00 | 435.18 | 2024-08-14 09:15:00 | 413.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-13 12:30:00 | 435.08 | 2024-08-14 09:15:00 | 413.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-13 14:15:00 | 435.18 | 2024-08-14 09:15:00 | 413.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-13 12:00:00 | 435.18 | 2024-09-04 13:15:00 | 391.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-13 12:30:00 | 435.08 | 2024-09-04 13:15:00 | 391.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-13 14:15:00 | 435.18 | 2024-09-04 13:15:00 | 391.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-14 09:15:00 | 406.98 | 2024-09-16 15:15:00 | 408.74 | PARTIAL | 0.50 | -0.43% |
| SELL | retest2 | 2024-08-14 09:15:00 | 406.98 | 2024-09-16 15:15:00 | 428.96 | STOP_HIT | 0.50 | -5.40% |
| SELL | retest2 | 2024-09-16 11:30:00 | 430.25 | 2024-09-16 15:15:00 | 408.14 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-09-16 11:30:00 | 430.25 | 2024-09-16 15:15:00 | 428.96 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2024-09-16 14:15:00 | 429.62 | 2024-09-16 15:15:00 | 407.51 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2024-09-16 14:15:00 | 429.62 | 2024-09-16 15:15:00 | 428.96 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2024-09-16 15:15:00 | 428.96 | 2024-09-18 09:15:00 | 435.60 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-10-23 12:30:00 | 447.15 | 2024-10-23 15:15:00 | 438.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-23 13:30:00 | 447.45 | 2024-10-23 15:15:00 | 438.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-11-06 12:30:00 | 450.00 | 2024-11-08 09:15:00 | 444.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-11-07 13:45:00 | 447.70 | 2024-11-08 09:15:00 | 444.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-01-07 13:00:00 | 494.60 | 2025-01-08 10:15:00 | 485.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-01-07 14:00:00 | 496.05 | 2025-01-08 10:15:00 | 485.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-03-17 10:30:00 | 396.00 | 2025-03-17 13:15:00 | 405.45 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-17 11:00:00 | 396.65 | 2025-03-17 13:15:00 | 405.45 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-07 15:15:00 | 447.50 | 2025-05-08 13:15:00 | 440.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-05-12 09:15:00 | 451.50 | 2025-05-19 09:15:00 | 496.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 09:15:00 | 514.25 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-09-29 12:15:00 | 509.50 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-30 09:15:00 | 508.00 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-30 09:45:00 | 509.15 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-13 14:00:00 | 517.30 | 2025-10-29 10:15:00 | 569.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 09:30:00 | 516.75 | 2025-10-29 10:15:00 | 568.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 11:15:00 | 515.55 | 2025-10-29 10:15:00 | 567.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 12:00:00 | 520.00 | 2025-10-29 10:15:00 | 561.22 | TARGET_HIT | 1.00 | 7.93% |
| BUY | retest2 | 2025-10-20 13:15:00 | 510.20 | 2025-10-29 10:15:00 | 563.59 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2025-10-21 13:45:00 | 512.35 | 2025-10-29 12:15:00 | 572.00 | TARGET_HIT | 1.00 | 11.64% |
| BUY | retest2 | 2025-11-17 13:30:00 | 510.00 | 2025-11-18 09:15:00 | 506.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-11-17 14:00:00 | 509.95 | 2025-11-18 09:15:00 | 506.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-26 11:00:00 | 533.25 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-11-26 12:00:00 | 538.15 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-11-28 09:30:00 | 532.15 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-11-28 11:15:00 | 532.70 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-12-03 09:30:00 | 525.25 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-03 11:30:00 | 524.95 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-04 09:45:00 | 526.65 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-12-04 13:15:00 | 525.25 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-22 09:15:00 | 533.90 | 2025-12-29 09:15:00 | 587.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-22 12:15:00 | 532.20 | 2025-12-29 09:15:00 | 585.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 09:45:00 | 532.70 | 2025-12-29 09:15:00 | 585.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-23 14:00:00 | 532.80 | 2026-02-01 13:15:00 | 522.95 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-10 12:15:00 | 564.85 | 2026-02-11 11:15:00 | 534.50 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2026-02-11 09:15:00 | 564.40 | 2026-02-11 11:15:00 | 534.50 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-02-19 15:00:00 | 545.50 | 2026-02-20 09:15:00 | 553.90 | STOP_HIT | 1.00 | -1.54% |
