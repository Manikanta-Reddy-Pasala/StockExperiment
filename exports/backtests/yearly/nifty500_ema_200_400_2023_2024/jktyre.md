# JK Tyre & Industries Ltd. (JKTYRE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 406.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 16 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 12 / 12
- **Target hits / Stop hits / Partials:** 2 / 14 / 8
- **Avg / median % per leg:** 1.07% / 1.93%
- **Sum % (uncompounded):** 25.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.91% | -23.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.91% | -23.3% |
| SELL (all) | 16 | 12 | 75.0% | 2 | 6 | 8 | 3.06% | 48.9% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.98% | 19.9% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 2 | 6 | 4 | 2.42% | 29.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.98% | 19.9% |
| retest2 (combined) | 20 | 8 | 40.0% | 2 | 14 | 4 | 0.29% | 5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 423.85 | 447.75 | 447.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 14:15:00 | 421.95 | 446.56 | 447.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 11:15:00 | 429.70 | 425.54 | 434.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-30 12:00:00 | 429.70 | 425.54 | 434.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 431.25 | 425.60 | 433.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:30:00 | 430.35 | 425.60 | 433.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 428.85 | 408.79 | 420.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 420.40 | 409.74 | 420.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 419.10 | 410.86 | 420.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 417.60 | 411.52 | 420.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 399.38 | 411.19 | 419.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 398.14 | 411.19 | 419.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 411.00 | 410.76 | 418.97 | SL hit (close>ema200) qty=0.50 sl=410.76 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 461.75 | 415.78 | 415.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 465.15 | 426.02 | 421.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 433.50 | 438.25 | 429.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:00:00 | 433.50 | 438.25 | 429.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 428.20 | 438.10 | 429.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 428.10 | 438.10 | 429.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 431.85 | 438.03 | 429.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 430.00 | 438.03 | 429.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 429.05 | 437.94 | 429.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 432.00 | 437.94 | 429.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 432.20 | 437.89 | 429.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 439.65 | 437.89 | 429.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 433.80 | 439.91 | 432.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:00:00 | 433.00 | 439.76 | 432.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:45:00 | 432.35 | 439.61 | 432.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 431.60 | 439.53 | 432.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 417.45 | 439.53 | 432.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 422.30 | 439.36 | 432.31 | SL hit (close<static) qty=1.00 sl=426.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 391.75 | 426.93 | 426.96 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 424.05 | 421.52 | 421.52 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 419.00 | 421.50 | 421.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 417.60 | 421.40 | 421.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 09:15:00 | 421.40 | 421.32 | 421.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 421.40 | 421.32 | 421.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 421.40 | 421.32 | 421.42 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 13:15:00 | 434.00 | 421.58 | 421.54 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 392.45 | 421.75 | 421.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 388.95 | 421.42 | 421.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 400.15 | 400.12 | 407.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 400.15 | 400.12 | 407.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 396.60 | 382.30 | 392.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 395.65 | 382.30 | 392.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 395.10 | 382.43 | 392.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 393.25 | 383.06 | 392.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 398.60 | 383.30 | 392.16 | SL hit (close>static) qty=1.00 sl=396.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 332.10 | 305.46 | 305.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 335.00 | 306.87 | 306.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 356.75 | 360.47 | 343.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 356.85 | 360.47 | 343.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 348.90 | 360.20 | 345.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 356.85 | 358.43 | 345.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 354.65 | 358.16 | 346.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:45:00 | 354.50 | 363.78 | 355.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 355.30 | 363.68 | 355.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 352.70 | 363.47 | 355.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 351.75 | 363.47 | 355.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 355.50 | 362.82 | 355.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 355.50 | 362.82 | 355.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 353.45 | 362.73 | 355.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 353.55 | 362.73 | 355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 353.55 | 362.64 | 355.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 353.55 | 362.64 | 355.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 353.85 | 362.39 | 355.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 351.95 | 362.39 | 355.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 344.75 | 361.04 | 355.03 | SL hit (close<static) qty=1.00 sl=345.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 321.10 | 350.35 | 350.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 318.55 | 348.64 | 349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:15:00 | 346.60 | 332.50 | 338.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 345.60 | 332.63 | 338.81 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 369.00 | 343.16 | 343.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 370.10 | 343.68 | 343.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 357.40 | 358.68 | 352.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 357.40 | 358.68 | 352.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 514.70 | 539.83 | 516.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 514.70 | 539.83 | 516.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 508.15 | 539.51 | 516.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 508.15 | 539.51 | 516.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 426.15 | 499.17 | 499.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 418.50 | 495.62 | 497.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 433.70 | 432.33 | 455.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 418.25 | 432.35 | 455.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 430.85 | 431.62 | 454.11 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:45:00 | 431.55 | 431.62 | 454.00 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 13:15:00 | 431.20 | 430.80 | 451.95 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.31 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.97 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.64 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 397.34 | 427.79 | 446.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 09:15:00 | 420.40 | 2024-05-31 09:15:00 | 399.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:45:00 | 419.10 | 2024-05-31 09:15:00 | 398.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 09:15:00 | 420.40 | 2024-06-03 09:15:00 | 411.00 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2024-05-27 09:45:00 | 419.10 | 2024-06-03 09:15:00 | 411.00 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2024-05-28 09:15:00 | 417.60 | 2024-06-04 09:15:00 | 396.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:15:00 | 417.60 | 2024-06-04 10:15:00 | 375.84 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-24 09:15:00 | 439.65 | 2024-08-05 09:15:00 | 422.30 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-08-02 10:30:00 | 433.80 | 2024-08-05 09:15:00 | 422.30 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-08-02 13:00:00 | 433.00 | 2024-08-05 09:15:00 | 422.30 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-08-02 14:45:00 | 432.35 | 2024-08-05 09:15:00 | 422.30 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-12-05 09:15:00 | 393.25 | 2024-12-05 10:15:00 | 398.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-05 12:45:00 | 394.00 | 2024-12-09 09:15:00 | 411.90 | STOP_HIT | 1.00 | -4.54% |
| SELL | retest2 | 2024-12-05 14:30:00 | 393.55 | 2024-12-09 09:15:00 | 411.90 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-12-06 09:45:00 | 393.90 | 2024-12-09 09:15:00 | 411.90 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2024-12-20 12:15:00 | 393.90 | 2025-01-06 12:15:00 | 374.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 393.90 | 2025-01-13 09:15:00 | 354.51 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-25 09:45:00 | 356.85 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-06-26 13:00:00 | 354.65 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-07-22 14:45:00 | 354.50 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-23 09:15:00 | 355.30 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest1 | 2026-04-13 09:15:00 | 418.25 | 2026-04-24 09:15:00 | 409.31 | PARTIAL | 0.50 | 2.14% |
| SELL | retest1 | 2026-04-15 11:00:00 | 430.85 | 2026-04-24 09:15:00 | 409.97 | PARTIAL | 0.50 | 4.85% |
| SELL | retest1 | 2026-04-15 11:45:00 | 431.55 | 2026-04-24 09:15:00 | 409.64 | PARTIAL | 0.50 | 5.08% |
| SELL | retest1 | 2026-04-17 13:15:00 | 431.20 | 2026-04-24 13:15:00 | 397.34 | PARTIAL | 0.50 | 7.85% |
