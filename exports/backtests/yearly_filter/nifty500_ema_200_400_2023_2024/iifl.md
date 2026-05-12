# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 460.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 1 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 12 |
| TARGET_HIT | 3 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 27
- **Target hits / Stop hits / Partials:** 3 / 34 / 12
- **Avg / median % per leg:** 0.84% / -0.61%
- **Sum % (uncompounded):** 41.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.61% | -20.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.61% | -20.9% |
| SELL (all) | 41 | 22 | 53.7% | 3 | 26 | 12 | 1.51% | 61.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 22 | 53.7% | 3 | 26 | 12 | 1.51% | 61.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 22 | 44.9% | 3 | 34 | 12 | 0.84% | 41.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 582.87 | 601.67 | 601.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 12:15:00 | 581.17 | 601.18 | 601.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 13:15:00 | 591.07 | 587.96 | 594.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-16 14:00:00 | 591.07 | 587.96 | 594.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 591.51 | 588.00 | 594.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 15:15:00 | 583.30 | 588.56 | 593.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 14:00:00 | 582.38 | 588.39 | 593.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 14:45:00 | 582.77 | 588.37 | 593.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 15:15:00 | 582.91 | 588.37 | 593.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 590.93 | 587.83 | 592.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 15:00:00 | 590.93 | 587.83 | 592.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 591.07 | 587.86 | 592.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 589.76 | 587.86 | 592.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 597.39 | 587.96 | 592.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-27 09:15:00 | 597.39 | 587.96 | 592.88 | SL hit (close>static) qty=1.00 sl=597.34 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 481.90 | 430.33 | 430.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 482.05 | 434.19 | 432.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 475.70 | 481.15 | 463.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 475.70 | 481.15 | 463.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 461.10 | 480.61 | 464.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 461.10 | 480.61 | 464.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 464.55 | 480.45 | 464.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 11:45:00 | 471.60 | 480.38 | 464.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:45:00 | 469.85 | 479.79 | 464.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 469.75 | 479.03 | 464.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 450.80 | 478.38 | 464.40 | SL hit (close<static) qty=1.00 sl=460.95 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 427.25 | 456.45 | 456.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 425.85 | 456.15 | 456.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 14:15:00 | 452.05 | 443.04 | 449.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 452.05 | 443.04 | 449.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 452.05 | 443.04 | 449.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:45:00 | 452.55 | 443.04 | 449.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 457.50 | 443.18 | 449.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 462.80 | 443.18 | 449.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 451.60 | 450.28 | 451.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 453.70 | 450.28 | 451.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 450.35 | 450.28 | 451.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:30:00 | 447.70 | 450.74 | 451.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 13:15:00 | 460.90 | 450.80 | 451.99 | SL hit (close>static) qty=1.00 sl=456.15 alert=retest2 |

### Cycle 4 — BUY (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 15:15:00 | 472.70 | 453.24 | 453.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 474.90 | 453.46 | 453.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 483.20 | 483.96 | 471.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 12:45:00 | 482.60 | 483.96 | 471.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 464.30 | 483.65 | 472.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 464.30 | 483.65 | 472.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 465.30 | 483.47 | 471.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:15:00 | 463.40 | 483.47 | 471.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 468.45 | 482.17 | 471.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 464.50 | 482.17 | 471.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 459.00 | 478.97 | 470.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 457.20 | 478.97 | 470.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 463.75 | 472.70 | 468.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 463.75 | 472.70 | 468.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 457.10 | 472.46 | 468.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 457.10 | 472.46 | 468.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 466.20 | 471.99 | 468.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 465.60 | 471.99 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 469.95 | 471.94 | 468.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 473.10 | 471.84 | 468.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 467.55 | 471.80 | 468.30 | SL hit (close<static) qty=1.00 sl=468.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 420.65 | 465.91 | 465.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 417.50 | 464.98 | 465.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 447.65 | 445.59 | 454.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 447.65 | 445.59 | 454.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 454.10 | 445.67 | 454.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 454.10 | 445.67 | 454.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 441.80 | 445.63 | 454.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:30:00 | 439.50 | 446.97 | 453.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 439.15 | 446.84 | 453.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 439.75 | 446.62 | 453.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 417.52 | 443.11 | 450.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 417.19 | 443.11 | 450.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 417.76 | 443.11 | 450.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-02 15:15:00 | 430.00 | 430.00 | 440.73 | SL hit (close>ema200) qty=0.50 sl=430.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 378.75 | 348.46 | 348.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 387.70 | 351.46 | 350.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 498.55 | 500.73 | 469.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 498.55 | 500.73 | 469.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 471.75 | 500.20 | 472.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 458.00 | 500.20 | 472.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 455.50 | 499.76 | 472.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 449.35 | 499.76 | 472.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 447.70 | 499.24 | 472.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 447.70 | 499.24 | 472.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 465.85 | 470.99 | 464.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 465.85 | 470.99 | 464.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 461.85 | 470.89 | 464.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 461.85 | 470.89 | 464.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 463.95 | 470.83 | 464.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 461.95 | 470.83 | 464.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 454.40 | 469.31 | 464.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 454.40 | 469.31 | 464.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 437.90 | 459.83 | 459.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 435.55 | 458.34 | 459.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 451.75 | 450.88 | 454.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:00:00 | 451.75 | 450.88 | 454.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 449.55 | 450.87 | 454.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 454.25 | 450.87 | 454.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 454.30 | 450.32 | 454.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 454.30 | 450.32 | 454.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 448.00 | 450.30 | 454.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 445.45 | 450.74 | 453.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:00:00 | 446.00 | 450.69 | 453.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 447.25 | 450.62 | 453.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 423.18 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 423.70 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 424.89 | 448.36 | 452.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 448.85 | 446.92 | 451.42 | SL hit (close>ema200) qty=0.50 sl=446.92 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 495.45 | 454.74 | 454.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 463.17 | 459.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 10:15:00 | 611.00 | 613.45 | 582.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:45:00 | 611.85 | 613.45 | 582.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 593.30 | 613.40 | 583.64 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 503.60 | 564.07 | 564.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 502.85 | 562.92 | 563.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 469.20 | 465.09 | 489.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 469.20 | 465.09 | 489.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-02-20 15:15:00 | 583.30 | 2024-02-27 09:15:00 | 597.39 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-02-21 14:00:00 | 582.38 | 2024-02-27 09:15:00 | 597.39 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-02-21 14:45:00 | 582.77 | 2024-02-27 09:15:00 | 597.39 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-02-21 15:15:00 | 582.91 | 2024-02-27 09:15:00 | 597.39 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-02-27 13:45:00 | 592.43 | 2024-02-28 10:15:00 | 562.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 13:45:00 | 592.43 | 2024-03-01 10:15:00 | 588.84 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-03-01 11:30:00 | 590.59 | 2024-03-01 13:15:00 | 599.48 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-03-04 09:45:00 | 591.85 | 2024-03-05 09:15:00 | 532.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-19 11:45:00 | 471.60 | 2024-07-23 12:15:00 | 450.80 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2024-07-22 09:45:00 | 469.85 | 2024-07-23 12:15:00 | 450.80 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-07-23 09:15:00 | 469.75 | 2024-07-23 12:15:00 | 450.80 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-08-01 10:15:00 | 474.25 | 2024-08-01 12:15:00 | 458.55 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2024-09-02 11:30:00 | 447.70 | 2024-09-02 13:15:00 | 460.90 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-10-14 09:15:00 | 473.10 | 2024-10-14 09:15:00 | 467.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-10-14 13:45:00 | 470.75 | 2024-10-15 12:15:00 | 468.15 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-14 14:30:00 | 471.00 | 2024-10-15 12:15:00 | 468.15 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-10-15 14:00:00 | 475.95 | 2024-10-17 09:15:00 | 463.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-11-11 09:30:00 | 439.50 | 2024-11-18 09:15:00 | 417.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 439.15 | 2024-11-18 09:15:00 | 417.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 439.75 | 2024-11-18 09:15:00 | 417.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:30:00 | 439.50 | 2024-12-02 15:15:00 | 430.00 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2024-11-11 15:15:00 | 439.15 | 2024-12-02 15:15:00 | 430.00 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2024-11-12 12:15:00 | 439.75 | 2024-12-02 15:15:00 | 430.00 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-12-04 10:30:00 | 437.45 | 2024-12-06 12:15:00 | 441.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-05 10:45:00 | 433.95 | 2024-12-06 12:15:00 | 441.85 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-12-05 11:15:00 | 434.40 | 2024-12-06 12:15:00 | 441.85 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-12-05 12:15:00 | 434.45 | 2024-12-06 12:15:00 | 441.85 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-05 14:30:00 | 434.30 | 2024-12-10 14:15:00 | 444.15 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-12-06 14:15:00 | 437.10 | 2024-12-10 14:15:00 | 444.15 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-12-10 13:30:00 | 438.85 | 2024-12-10 15:15:00 | 447.10 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-10 14:45:00 | 438.80 | 2024-12-13 10:15:00 | 415.58 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2024-12-12 09:45:00 | 438.40 | 2024-12-13 10:15:00 | 416.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 14:45:00 | 438.80 | 2024-12-13 12:15:00 | 433.05 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-12-12 09:45:00 | 438.40 | 2024-12-13 12:15:00 | 433.05 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-01-03 11:30:00 | 432.00 | 2025-01-06 10:15:00 | 410.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 11:30:00 | 432.00 | 2025-01-07 15:15:00 | 422.55 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2025-01-09 09:15:00 | 428.30 | 2025-01-13 09:15:00 | 409.59 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2025-01-09 11:00:00 | 431.15 | 2025-01-13 11:15:00 | 406.88 | PARTIAL | 0.50 | 5.63% |
| SELL | retest2 | 2025-01-09 09:15:00 | 428.30 | 2025-01-22 09:15:00 | 385.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 431.15 | 2025-01-22 09:15:00 | 388.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 09:30:00 | 445.45 | 2025-09-26 13:15:00 | 423.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 446.00 | 2025-09-26 13:15:00 | 423.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:30:00 | 447.25 | 2025-09-26 13:15:00 | 424.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:30:00 | 445.45 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.76% |
| SELL | retest2 | 2025-09-23 11:00:00 | 446.00 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.64% |
| SELL | retest2 | 2025-09-24 10:30:00 | 447.25 | 2025-09-30 09:15:00 | 448.85 | STOP_HIT | 0.50 | -0.36% |
| SELL | retest2 | 2025-09-30 11:00:00 | 447.20 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-10-01 09:45:00 | 447.40 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-10-01 14:30:00 | 449.45 | 2025-10-03 09:15:00 | 465.90 | STOP_HIT | 1.00 | -3.66% |
