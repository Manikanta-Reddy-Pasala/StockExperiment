# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 836.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 251 |
| ALERT1 | 155 |
| ALERT2 | 153 |
| ALERT2_SKIP | 74 |
| ALERT3 | 434 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 180 |
| PARTIAL | 22 |
| TARGET_HIT | 11 |
| STOP_HIT | 176 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 209 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 77 / 132
- **Target hits / Stop hits / Partials:** 11 / 176 / 22
- **Avg / median % per leg:** 0.55% / -0.73%
- **Sum % (uncompounded):** 114.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 94 | 24 | 25.5% | 10 | 82 | 2 | 0.33% | 30.8% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 0 | 7 | 2 | 1.10% | 9.9% |
| BUY @ 3rd Alert (retest2) | 85 | 18 | 21.2% | 10 | 75 | 0 | 0.25% | 20.9% |
| SELL (all) | 115 | 53 | 46.1% | 1 | 94 | 20 | 0.73% | 84.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 115 | 53 | 46.1% | 1 | 94 | 20 | 0.73% | 84.1% |
| retest1 (combined) | 9 | 6 | 66.7% | 0 | 7 | 2 | 1.10% | 9.9% |
| retest2 (combined) | 200 | 71 | 35.5% | 11 | 169 | 20 | 0.53% | 105.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 14:15:00 | 483.50 | 498.71 | 499.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 472.50 | 483.34 | 490.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 15:15:00 | 452.85 | 452.72 | 457.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 09:30:00 | 451.10 | 452.35 | 457.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 460.30 | 453.87 | 455.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 10:00:00 | 460.30 | 453.87 | 455.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 461.45 | 455.38 | 456.00 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 12:15:00 | 458.40 | 456.74 | 456.56 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 455.20 | 456.43 | 456.43 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 14:15:00 | 458.15 | 456.78 | 456.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 461.85 | 457.67 | 457.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 14:15:00 | 459.30 | 459.93 | 458.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 459.30 | 459.93 | 458.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 459.30 | 459.93 | 458.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 459.30 | 459.93 | 458.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 456.85 | 459.49 | 458.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 457.10 | 459.49 | 458.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 457.60 | 459.11 | 458.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 11:30:00 | 459.20 | 459.17 | 458.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:00:00 | 458.60 | 459.08 | 458.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 15:15:00 | 459.00 | 458.74 | 458.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 11:15:00 | 458.45 | 458.84 | 458.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 11:15:00 | 456.65 | 458.40 | 458.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 11:15:00 | 456.65 | 458.40 | 458.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 09:15:00 | 453.40 | 456.93 | 457.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 452.05 | 451.86 | 453.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-31 15:00:00 | 452.05 | 451.86 | 453.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 452.50 | 451.99 | 453.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:45:00 | 454.45 | 452.26 | 453.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 453.80 | 452.57 | 453.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:30:00 | 454.00 | 452.57 | 453.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 11:15:00 | 453.90 | 452.83 | 453.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:00:00 | 453.90 | 452.83 | 453.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 454.60 | 453.19 | 453.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:30:00 | 454.20 | 453.19 | 453.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 13:15:00 | 457.70 | 454.09 | 454.05 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 452.85 | 454.07 | 454.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 14:15:00 | 452.25 | 453.60 | 453.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 454.35 | 453.57 | 453.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 454.35 | 453.57 | 453.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 454.35 | 453.57 | 453.85 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 10:15:00 | 456.00 | 454.06 | 454.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 12:15:00 | 459.20 | 455.45 | 454.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 13:15:00 | 471.45 | 472.89 | 470.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 14:00:00 | 471.45 | 472.89 | 470.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 471.70 | 472.65 | 470.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:45:00 | 468.70 | 472.65 | 470.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 470.90 | 472.34 | 470.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 468.10 | 472.34 | 470.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 470.55 | 471.98 | 470.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 10:45:00 | 471.15 | 471.98 | 470.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 468.55 | 471.30 | 470.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:00:00 | 468.55 | 471.30 | 470.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 468.80 | 470.80 | 470.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 468.80 | 470.80 | 470.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 15:15:00 | 467.25 | 469.92 | 470.06 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 09:15:00 | 473.00 | 470.54 | 470.33 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 12:15:00 | 468.30 | 470.06 | 470.17 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 471.95 | 470.40 | 470.24 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 14:15:00 | 467.95 | 469.75 | 469.98 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 12:15:00 | 479.05 | 471.53 | 470.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 13:15:00 | 479.65 | 473.16 | 471.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 14:15:00 | 476.90 | 477.86 | 475.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 15:00:00 | 476.90 | 477.86 | 475.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 475.50 | 477.38 | 475.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 478.40 | 477.38 | 475.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 10:15:00 | 488.25 | 489.06 | 489.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 488.25 | 489.06 | 489.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 482.50 | 487.75 | 488.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 15:15:00 | 490.00 | 484.65 | 486.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 15:15:00 | 490.00 | 484.65 | 486.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 490.00 | 484.65 | 486.42 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 497.30 | 488.97 | 488.06 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 11:15:00 | 481.95 | 487.16 | 487.87 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 488.55 | 485.35 | 485.26 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 13:15:00 | 482.80 | 484.75 | 485.00 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 488.60 | 485.19 | 485.09 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 13:15:00 | 483.40 | 485.14 | 485.18 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 14:15:00 | 486.65 | 485.44 | 485.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 09:15:00 | 505.40 | 489.52 | 487.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 15:15:00 | 511.00 | 512.12 | 505.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:15:00 | 515.05 | 512.12 | 505.54 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:15:00 | 516.05 | 512.24 | 506.19 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 13:15:00 | 540.80 | 520.27 | 512.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 13:15:00 | 541.85 | 520.27 | 512.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-07-06 12:15:00 | 529.30 | 529.36 | 521.24 | SL hit (close<ema200) qty=0.50 sl=529.36 alert=retest1 |

### Cycle 23 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 525.05 | 530.75 | 530.81 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 15:15:00 | 533.00 | 530.63 | 530.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 537.45 | 531.99 | 531.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 10:15:00 | 531.10 | 531.81 | 531.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 10:15:00 | 531.10 | 531.81 | 531.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 531.10 | 531.81 | 531.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:45:00 | 529.80 | 531.81 | 531.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 529.50 | 531.35 | 531.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:00:00 | 529.50 | 531.35 | 531.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 529.80 | 531.04 | 530.90 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 525.05 | 529.84 | 530.37 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 533.85 | 530.73 | 530.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 11:15:00 | 539.00 | 532.38 | 531.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 12:15:00 | 538.30 | 539.59 | 536.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 13:00:00 | 538.30 | 539.59 | 536.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 535.35 | 538.74 | 536.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 533.35 | 538.74 | 536.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 533.25 | 537.64 | 536.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 533.25 | 537.64 | 536.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 533.85 | 536.88 | 535.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 531.75 | 535.77 | 535.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 10:15:00 | 530.00 | 534.61 | 534.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 11:15:00 | 528.55 | 533.40 | 534.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 13:15:00 | 528.15 | 527.58 | 529.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-19 14:00:00 | 528.15 | 527.58 | 529.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 533.35 | 528.73 | 530.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 15:00:00 | 533.35 | 528.73 | 530.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 15:15:00 | 536.90 | 530.36 | 530.86 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 537.90 | 531.87 | 531.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 10:15:00 | 557.45 | 536.99 | 533.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 09:15:00 | 551.25 | 551.63 | 544.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 10:00:00 | 551.25 | 551.63 | 544.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 555.05 | 564.03 | 560.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 13:45:00 | 568.25 | 563.96 | 561.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 11:15:00 | 555.10 | 559.96 | 560.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 555.10 | 559.96 | 560.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 13:15:00 | 550.30 | 555.55 | 557.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 559.00 | 556.24 | 557.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 14:15:00 | 559.00 | 556.24 | 557.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 559.00 | 556.24 | 557.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 559.00 | 556.24 | 557.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 555.00 | 555.99 | 557.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 562.15 | 555.99 | 557.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 564.25 | 557.64 | 557.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 564.25 | 557.64 | 557.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 561.95 | 558.51 | 558.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 12:15:00 | 573.00 | 562.12 | 560.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 552.60 | 564.30 | 562.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 552.60 | 564.30 | 562.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 552.60 | 564.30 | 562.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:45:00 | 555.00 | 564.30 | 562.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 553.95 | 562.23 | 561.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:45:00 | 554.95 | 562.23 | 561.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 12:15:00 | 547.90 | 559.36 | 560.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 541.70 | 554.52 | 557.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 541.50 | 540.89 | 547.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 10:00:00 | 541.50 | 540.89 | 547.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 537.80 | 534.36 | 539.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:30:00 | 535.25 | 535.02 | 538.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:15:00 | 534.45 | 535.02 | 537.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 10:15:00 | 533.90 | 530.93 | 530.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 533.90 | 530.93 | 530.64 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 528.25 | 530.48 | 530.52 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 14:15:00 | 534.00 | 531.19 | 530.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 15:15:00 | 535.00 | 531.95 | 531.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 10:15:00 | 530.95 | 531.88 | 531.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 10:15:00 | 530.95 | 531.88 | 531.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 530.95 | 531.88 | 531.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:00:00 | 530.95 | 531.88 | 531.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 531.40 | 531.78 | 531.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:45:00 | 530.65 | 531.78 | 531.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 530.05 | 531.44 | 531.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:00:00 | 530.05 | 531.44 | 531.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 531.00 | 531.35 | 531.19 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 517.70 | 528.62 | 529.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 15:15:00 | 509.70 | 516.28 | 522.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 11:15:00 | 503.95 | 503.09 | 506.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-18 11:45:00 | 503.00 | 503.09 | 506.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 499.50 | 496.15 | 499.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:00:00 | 499.50 | 496.15 | 499.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 500.00 | 496.92 | 499.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:30:00 | 501.00 | 496.92 | 499.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 502.75 | 498.08 | 499.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:45:00 | 503.20 | 498.08 | 499.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 14:15:00 | 507.20 | 501.11 | 500.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 513.70 | 505.52 | 503.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 586.50 | 588.77 | 574.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 11:00:00 | 586.50 | 588.77 | 574.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 586.30 | 587.90 | 585.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:45:00 | 584.00 | 587.90 | 585.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 585.60 | 587.44 | 585.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:30:00 | 587.25 | 587.44 | 585.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 587.00 | 587.35 | 585.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 584.75 | 587.35 | 585.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 575.60 | 585.00 | 584.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:00:00 | 575.60 | 585.00 | 584.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 10:15:00 | 574.95 | 582.99 | 583.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 13:15:00 | 572.65 | 578.42 | 581.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 10:15:00 | 557.85 | 557.50 | 563.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-07 11:00:00 | 557.85 | 557.50 | 563.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 566.00 | 559.20 | 563.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:00:00 | 566.00 | 559.20 | 563.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 583.00 | 563.96 | 565.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 13:00:00 | 583.00 | 563.96 | 565.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 595.75 | 570.32 | 568.42 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 15:15:00 | 575.00 | 578.15 | 578.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 559.20 | 574.36 | 576.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 565.00 | 558.75 | 563.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 565.00 | 558.75 | 563.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 565.00 | 558.75 | 563.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:15:00 | 564.00 | 558.75 | 563.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 564.30 | 559.86 | 563.39 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-09-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 13:15:00 | 575.90 | 566.09 | 565.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 581.15 | 572.99 | 569.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 615.30 | 624.50 | 614.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 615.30 | 624.50 | 614.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 615.30 | 624.50 | 614.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:00:00 | 615.30 | 624.50 | 614.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 615.00 | 622.60 | 614.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:30:00 | 614.35 | 622.60 | 614.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 613.50 | 620.78 | 614.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:30:00 | 614.25 | 620.78 | 614.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 607.95 | 618.21 | 614.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 606.25 | 618.21 | 614.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 635.15 | 618.37 | 615.01 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 10:15:00 | 608.00 | 615.28 | 615.76 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 622.30 | 616.56 | 616.06 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 612.25 | 615.44 | 615.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 11:15:00 | 610.55 | 614.46 | 615.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 14:15:00 | 615.30 | 614.02 | 614.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 14:15:00 | 615.30 | 614.02 | 614.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 615.30 | 614.02 | 614.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 615.30 | 614.02 | 614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 615.00 | 614.21 | 614.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 615.85 | 614.21 | 614.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 615.15 | 614.40 | 614.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:30:00 | 617.00 | 614.40 | 614.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 610.70 | 613.66 | 614.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 12:15:00 | 609.20 | 613.13 | 614.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 15:00:00 | 607.95 | 611.84 | 613.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 11:15:00 | 616.45 | 612.27 | 612.89 | SL hit (close>static) qty=1.00 sl=615.15 alert=retest2 |

### Cycle 44 — BUY (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 13:15:00 | 615.00 | 613.64 | 613.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 15:15:00 | 620.55 | 615.23 | 614.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 13:15:00 | 627.55 | 627.59 | 623.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-03 13:45:00 | 625.05 | 627.59 | 623.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 617.30 | 625.43 | 623.58 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 611.75 | 621.26 | 621.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 608.45 | 618.69 | 620.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 615.55 | 613.45 | 616.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 13:45:00 | 615.25 | 613.45 | 616.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 617.50 | 614.26 | 616.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:45:00 | 617.35 | 614.26 | 616.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 617.10 | 614.82 | 616.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 620.80 | 614.82 | 616.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 620.50 | 615.96 | 616.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:45:00 | 621.80 | 615.96 | 616.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 616.20 | 616.30 | 616.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:00:00 | 616.20 | 616.30 | 616.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 616.85 | 616.41 | 616.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:15:00 | 617.10 | 616.41 | 616.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 615.35 | 616.20 | 616.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 595.45 | 616.28 | 616.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 15:15:00 | 565.68 | 580.82 | 587.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-19 11:15:00 | 582.65 | 580.60 | 585.54 | SL hit (close>ema200) qty=0.50 sl=580.60 alert=retest2 |

### Cycle 46 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 563.00 | 557.06 | 556.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 565.55 | 559.32 | 557.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 560.65 | 561.45 | 559.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 560.65 | 561.45 | 559.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 560.65 | 561.45 | 559.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:15:00 | 558.00 | 561.45 | 559.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 558.00 | 560.76 | 559.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 10:00:00 | 563.90 | 561.39 | 559.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 12:15:00 | 554.30 | 559.57 | 559.40 | SL hit (close<static) qty=1.00 sl=556.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 553.10 | 558.28 | 558.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 15:15:00 | 551.25 | 556.54 | 557.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 560.45 | 557.32 | 558.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 560.45 | 557.32 | 558.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 560.45 | 557.32 | 558.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 562.05 | 557.32 | 558.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 568.00 | 559.46 | 559.04 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 555.50 | 559.92 | 560.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 553.80 | 558.70 | 559.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 556.45 | 555.59 | 557.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 556.45 | 555.59 | 557.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 556.45 | 555.59 | 557.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:30:00 | 554.90 | 555.59 | 557.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 554.85 | 555.44 | 557.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 11:15:00 | 552.55 | 555.44 | 557.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 09:15:00 | 552.90 | 554.49 | 555.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 10:15:00 | 559.00 | 553.73 | 554.15 | SL hit (close>static) qty=1.00 sl=557.95 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 565.40 | 556.07 | 555.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 584.05 | 563.65 | 560.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 10:15:00 | 577.55 | 577.88 | 571.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-15 11:00:00 | 577.55 | 577.88 | 571.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 12:15:00 | 572.75 | 575.99 | 572.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 12:30:00 | 570.55 | 575.99 | 572.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 571.40 | 575.07 | 571.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 14:00:00 | 571.40 | 575.07 | 571.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 14:15:00 | 571.90 | 574.44 | 571.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 09:15:00 | 575.35 | 574.25 | 572.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 11:30:00 | 575.80 | 574.21 | 572.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 12:15:00 | 570.50 | 573.47 | 572.40 | SL hit (close<static) qty=1.00 sl=570.55 alert=retest2 |

### Cycle 51 — SELL (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 15:15:00 | 569.00 | 571.25 | 571.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 564.75 | 567.57 | 569.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 15:15:00 | 566.80 | 565.77 | 567.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 09:15:00 | 566.35 | 565.77 | 567.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 562.90 | 565.20 | 567.06 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 571.25 | 567.72 | 567.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 572.00 | 568.57 | 567.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 15:15:00 | 575.00 | 575.42 | 573.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 15:15:00 | 575.00 | 575.42 | 573.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 575.00 | 575.42 | 573.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:30:00 | 582.40 | 576.44 | 574.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 12:15:00 | 568.05 | 576.19 | 574.82 | SL hit (close<static) qty=1.00 sl=573.00 alert=retest2 |

### Cycle 53 — SELL (started 2023-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 14:15:00 | 566.40 | 572.88 | 573.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 10:15:00 | 561.15 | 568.54 | 571.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 12:15:00 | 564.70 | 561.39 | 564.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 12:15:00 | 564.70 | 561.39 | 564.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 564.70 | 561.39 | 564.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 13:00:00 | 564.70 | 561.39 | 564.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 566.60 | 562.43 | 564.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 13:30:00 | 569.00 | 562.43 | 564.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 567.70 | 563.49 | 565.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 567.70 | 563.49 | 565.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 568.95 | 564.58 | 565.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 561.35 | 564.58 | 565.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 564.50 | 564.56 | 565.43 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 15:15:00 | 567.50 | 566.04 | 565.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 10:15:00 | 575.65 | 568.66 | 567.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 571.00 | 571.35 | 569.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 14:30:00 | 571.05 | 571.35 | 569.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 569.95 | 571.07 | 569.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 574.40 | 571.07 | 569.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 10:15:00 | 577.80 | 571.25 | 569.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 10:15:00 | 569.45 | 576.21 | 576.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 10:15:00 | 569.45 | 576.21 | 576.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 561.80 | 572.01 | 574.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 10:15:00 | 557.10 | 554.81 | 558.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 10:15:00 | 557.10 | 554.81 | 558.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 557.10 | 554.81 | 558.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:30:00 | 558.10 | 554.81 | 558.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 553.60 | 554.57 | 557.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:30:00 | 554.75 | 554.57 | 557.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 553.00 | 549.38 | 552.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:30:00 | 553.25 | 549.38 | 552.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 558.85 | 551.28 | 552.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 558.85 | 551.28 | 552.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 557.60 | 552.54 | 553.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 560.95 | 552.54 | 553.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 556.20 | 553.74 | 553.58 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 549.10 | 552.77 | 553.19 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 559.75 | 553.92 | 553.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 14:15:00 | 562.70 | 559.20 | 556.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 15:15:00 | 565.00 | 566.16 | 562.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:15:00 | 570.95 | 566.16 | 562.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 565.00 | 566.01 | 563.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:45:00 | 562.10 | 566.01 | 563.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 564.95 | 565.79 | 563.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:30:00 | 564.50 | 565.79 | 563.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 562.00 | 565.04 | 563.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-19 12:15:00 | 562.00 | 565.04 | 563.18 | SL hit (close<ema400) qty=1.00 sl=563.18 alert=retest1 |

### Cycle 59 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 551.40 | 562.12 | 562.87 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 561.50 | 561.13 | 561.12 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 559.20 | 560.95 | 561.05 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 563.00 | 561.10 | 561.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 15:15:00 | 563.95 | 561.67 | 561.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 09:15:00 | 561.55 | 561.65 | 561.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 10:00:00 | 561.55 | 561.65 | 561.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 63 — SELL (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 10:15:00 | 558.80 | 561.08 | 561.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 12:15:00 | 556.85 | 559.75 | 560.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 15:15:00 | 559.95 | 559.04 | 559.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 15:15:00 | 559.95 | 559.04 | 559.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 559.95 | 559.04 | 559.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:15:00 | 560.35 | 559.04 | 559.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 559.05 | 559.05 | 559.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:30:00 | 559.60 | 559.05 | 559.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 557.00 | 558.64 | 559.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:30:00 | 558.80 | 558.64 | 559.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 558.70 | 558.23 | 559.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 15:00:00 | 558.70 | 558.23 | 559.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 560.00 | 558.58 | 559.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-28 09:15:00 | 566.00 | 558.58 | 559.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 09:15:00 | 568.30 | 560.53 | 559.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 11:15:00 | 583.00 | 576.51 | 571.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 13:15:00 | 577.00 | 577.13 | 572.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 14:00:00 | 577.00 | 577.13 | 572.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 569.75 | 575.66 | 572.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 569.75 | 575.66 | 572.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 572.00 | 574.92 | 572.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 573.00 | 574.92 | 572.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 11:15:00 | 564.20 | 570.89 | 570.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 564.20 | 570.89 | 570.97 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 14:15:00 | 581.75 | 572.82 | 571.74 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 10:15:00 | 570.60 | 575.33 | 575.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 568.10 | 572.35 | 574.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 09:15:00 | 557.75 | 555.75 | 560.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 10:15:00 | 558.70 | 555.75 | 560.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 558.40 | 557.06 | 559.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 558.40 | 557.06 | 559.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 559.65 | 557.73 | 559.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 562.65 | 557.73 | 559.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 557.20 | 557.62 | 559.26 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 13:15:00 | 563.70 | 560.59 | 560.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 569.05 | 563.61 | 561.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 565.75 | 567.93 | 565.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 565.75 | 567.93 | 565.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 565.75 | 567.93 | 565.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 565.75 | 567.93 | 565.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 563.80 | 567.10 | 565.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:00:00 | 563.80 | 567.10 | 565.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 563.80 | 566.44 | 565.19 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 15:15:00 | 563.00 | 564.53 | 564.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 559.00 | 562.82 | 563.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 15:15:00 | 554.90 | 554.18 | 557.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:15:00 | 553.60 | 554.18 | 557.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 556.05 | 552.74 | 554.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:00:00 | 554.15 | 553.02 | 554.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:30:00 | 555.00 | 553.02 | 554.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 596.65 | 556.81 | 554.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 596.65 | 556.81 | 554.00 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 549.50 | 569.59 | 570.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 11:15:00 | 538.70 | 560.06 | 565.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 530.30 | 526.91 | 533.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 530.30 | 526.91 | 533.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 530.30 | 526.91 | 533.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 10:15:00 | 527.90 | 526.91 | 533.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 13:45:00 | 527.55 | 528.42 | 532.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 13:00:00 | 526.40 | 527.42 | 529.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 14:15:00 | 534.50 | 528.79 | 528.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 14:15:00 | 534.50 | 528.79 | 528.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 09:15:00 | 546.40 | 534.04 | 531.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 15:15:00 | 546.90 | 546.99 | 542.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:15:00 | 553.00 | 546.99 | 542.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 545.90 | 549.25 | 545.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:00:00 | 545.90 | 549.25 | 545.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 538.40 | 547.08 | 545.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-08 14:15:00 | 538.40 | 547.08 | 545.16 | SL hit (close<ema400) qty=1.00 sl=545.16 alert=retest1 |

### Cycle 73 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 534.65 | 542.66 | 543.45 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 548.80 | 544.19 | 543.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 15:15:00 | 548.85 | 545.56 | 544.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 542.45 | 544.94 | 544.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 542.45 | 544.94 | 544.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 542.45 | 544.94 | 544.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:00:00 | 542.45 | 544.94 | 544.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 534.55 | 542.86 | 543.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 527.00 | 539.69 | 542.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 12:15:00 | 534.65 | 530.94 | 534.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 12:15:00 | 534.65 | 530.94 | 534.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 534.65 | 530.94 | 534.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 534.65 | 530.94 | 534.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 539.60 | 532.67 | 535.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:00:00 | 539.60 | 532.67 | 535.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 533.00 | 532.74 | 534.96 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 10:15:00 | 543.45 | 537.30 | 536.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 546.30 | 541.37 | 539.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 540.20 | 541.41 | 539.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 10:00:00 | 540.20 | 541.41 | 539.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 543.55 | 541.84 | 539.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:30:00 | 540.00 | 541.84 | 539.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 541.35 | 542.53 | 540.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:45:00 | 542.45 | 542.53 | 540.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 541.00 | 542.22 | 540.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:00:00 | 541.00 | 542.22 | 540.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 540.25 | 541.83 | 540.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 15:00:00 | 540.25 | 541.83 | 540.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 541.95 | 541.85 | 540.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:15:00 | 545.50 | 541.85 | 540.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 10:15:00 | 542.80 | 543.20 | 542.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 11:00:00 | 542.90 | 543.14 | 542.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 14:15:00 | 538.35 | 541.58 | 541.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 538.35 | 541.58 | 541.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 537.40 | 540.48 | 541.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 11:15:00 | 537.05 | 536.53 | 538.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-21 12:00:00 | 537.05 | 536.53 | 538.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 538.50 | 534.56 | 536.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 537.25 | 534.56 | 536.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 537.70 | 535.19 | 536.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 537.70 | 535.19 | 536.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 537.25 | 535.60 | 536.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 537.25 | 535.60 | 536.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 536.40 | 535.76 | 536.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:15:00 | 534.65 | 536.62 | 536.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 12:15:00 | 537.50 | 536.44 | 536.64 | SL hit (close>static) qty=1.00 sl=537.45 alert=retest2 |

### Cycle 78 — BUY (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 15:15:00 | 539.50 | 536.74 | 536.43 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 535.05 | 536.21 | 536.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 534.95 | 535.67 | 535.98 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 11:15:00 | 539.90 | 536.52 | 536.33 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 527.75 | 534.58 | 535.50 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 536.45 | 534.47 | 534.36 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 10:15:00 | 532.50 | 534.19 | 534.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 530.00 | 533.09 | 533.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 13:15:00 | 534.25 | 532.21 | 532.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 13:15:00 | 534.25 | 532.21 | 532.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 534.25 | 532.21 | 532.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:45:00 | 535.90 | 532.21 | 532.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 535.10 | 532.79 | 533.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:30:00 | 536.40 | 532.79 | 533.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 528.30 | 532.18 | 532.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 11:30:00 | 526.60 | 530.35 | 531.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 13:00:00 | 525.60 | 529.40 | 531.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 14:15:00 | 526.30 | 529.19 | 531.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:15:00 | 527.05 | 525.76 | 528.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 527.65 | 526.14 | 528.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:00:00 | 527.65 | 526.14 | 528.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 526.70 | 526.25 | 527.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 15:00:00 | 520.40 | 525.08 | 527.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 14:45:00 | 523.95 | 521.60 | 523.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 11:15:00 | 521.80 | 516.25 | 517.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 13:15:00 | 519.75 | 518.49 | 518.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 13:15:00 | 519.75 | 518.49 | 518.32 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 510.00 | 516.94 | 517.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 508.15 | 515.18 | 516.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 09:15:00 | 511.70 | 511.12 | 513.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 511.70 | 511.12 | 513.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 511.70 | 511.12 | 513.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 510.90 | 511.12 | 513.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 509.95 | 510.21 | 512.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:30:00 | 510.30 | 510.21 | 512.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 516.00 | 509.47 | 510.78 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 14:15:00 | 518.15 | 512.50 | 511.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 521.45 | 514.52 | 513.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 13:15:00 | 517.50 | 518.30 | 515.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-21 14:00:00 | 517.50 | 518.30 | 515.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 515.65 | 518.00 | 516.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:15:00 | 521.00 | 518.00 | 516.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 524.85 | 519.37 | 516.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 10:15:00 | 526.55 | 519.37 | 516.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 11:15:00 | 526.05 | 520.68 | 517.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 13:15:00 | 526.75 | 522.29 | 519.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-05 09:15:00 | 579.21 | 576.24 | 566.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 588.65 | 595.04 | 595.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 583.30 | 587.91 | 590.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 590.45 | 588.42 | 590.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 10:15:00 | 590.45 | 588.42 | 590.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 590.45 | 588.42 | 590.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 591.65 | 588.42 | 590.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 584.45 | 587.62 | 590.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 589.00 | 587.62 | 590.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 587.70 | 586.64 | 588.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 586.85 | 586.64 | 588.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 585.60 | 586.43 | 588.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:30:00 | 583.20 | 585.62 | 587.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:30:00 | 582.55 | 585.16 | 587.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:00:00 | 583.50 | 584.83 | 587.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 11:15:00 | 588.95 | 579.37 | 579.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 588.95 | 579.37 | 579.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 13:15:00 | 602.80 | 585.93 | 582.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 602.10 | 602.21 | 595.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:30:00 | 598.60 | 602.54 | 596.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 595.65 | 600.76 | 596.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 595.65 | 600.76 | 596.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 595.25 | 599.66 | 596.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 594.60 | 599.66 | 596.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 598.15 | 598.21 | 596.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 597.00 | 598.21 | 596.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 598.00 | 598.17 | 596.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:30:00 | 605.95 | 599.60 | 597.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 13:45:00 | 605.55 | 601.58 | 598.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:45:00 | 603.80 | 606.77 | 603.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 604.10 | 605.92 | 603.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 603.35 | 605.41 | 603.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 603.35 | 605.41 | 603.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 601.30 | 604.59 | 603.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:00:00 | 601.30 | 604.59 | 603.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 600.00 | 603.67 | 603.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:45:00 | 600.00 | 603.67 | 603.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 603.30 | 604.29 | 603.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:00:00 | 603.30 | 604.29 | 603.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 602.90 | 604.01 | 603.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 602.90 | 604.01 | 603.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 602.00 | 603.61 | 603.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 601.50 | 603.61 | 603.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 600.30 | 602.95 | 603.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 600.30 | 602.95 | 603.07 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 14:15:00 | 609.05 | 603.12 | 602.98 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 601.25 | 602.75 | 602.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 600.00 | 602.16 | 602.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 12:15:00 | 604.35 | 601.92 | 602.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 12:15:00 | 604.35 | 601.92 | 602.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 604.35 | 601.92 | 602.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:30:00 | 608.25 | 601.92 | 602.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 602.20 | 601.98 | 602.31 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 10:15:00 | 641.20 | 609.82 | 605.79 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 613.00 | 624.59 | 626.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 11:15:00 | 604.95 | 617.08 | 621.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 10:15:00 | 610.00 | 599.25 | 605.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 10:15:00 | 610.00 | 599.25 | 605.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 610.00 | 599.25 | 605.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 610.00 | 599.25 | 605.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 594.20 | 598.24 | 604.59 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 14:15:00 | 612.60 | 604.33 | 604.00 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 604.10 | 607.07 | 607.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 599.75 | 604.85 | 606.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 594.75 | 593.74 | 596.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:45:00 | 593.95 | 593.74 | 596.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 601.45 | 595.28 | 596.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 601.45 | 595.28 | 596.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 606.55 | 597.53 | 597.61 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 604.50 | 598.93 | 598.24 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 595.35 | 597.84 | 598.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 15:15:00 | 593.00 | 595.65 | 596.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 586.55 | 584.76 | 588.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 11:15:00 | 586.55 | 584.76 | 588.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 586.55 | 584.76 | 588.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 586.55 | 584.76 | 588.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 588.40 | 585.49 | 588.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 588.40 | 585.49 | 588.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 589.00 | 586.19 | 588.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 589.05 | 586.19 | 588.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 589.90 | 586.93 | 588.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 589.90 | 586.93 | 588.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 589.95 | 587.54 | 588.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 587.00 | 587.54 | 588.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 590.50 | 588.13 | 588.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:15:00 | 597.00 | 588.13 | 588.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 593.25 | 589.15 | 589.34 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 592.15 | 589.75 | 589.60 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 588.75 | 589.67 | 589.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 10:15:00 | 586.55 | 588.83 | 589.27 | Break + close below crossover candle low |

### Cycle 100 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 595.85 | 590.24 | 589.86 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 12:15:00 | 587.05 | 589.60 | 589.61 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 592.00 | 590.08 | 589.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 594.80 | 591.02 | 590.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 585.50 | 590.55 | 590.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 585.50 | 590.55 | 590.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 585.50 | 590.55 | 590.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 576.95 | 590.55 | 590.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 569.00 | 586.24 | 588.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 554.75 | 579.95 | 585.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 15:15:00 | 580.00 | 577.13 | 581.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 09:15:00 | 573.70 | 577.13 | 581.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 573.20 | 576.34 | 581.06 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 595.00 | 583.01 | 581.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 608.45 | 593.41 | 588.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 11:15:00 | 617.20 | 617.65 | 611.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:00:00 | 617.20 | 617.65 | 611.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 612.95 | 616.02 | 613.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 612.95 | 616.02 | 613.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 614.75 | 615.77 | 613.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 616.05 | 615.77 | 613.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 622.70 | 615.96 | 614.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 09:15:00 | 677.65 | 656.10 | 641.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 686.00 | 692.80 | 693.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 11:15:00 | 681.50 | 687.11 | 689.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 682.05 | 681.20 | 685.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:00:00 | 682.05 | 681.20 | 685.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 680.30 | 676.33 | 680.22 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 696.40 | 683.37 | 682.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 700.00 | 690.27 | 687.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 692.05 | 693.28 | 690.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 14:00:00 | 692.05 | 693.28 | 690.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 689.70 | 692.56 | 690.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 689.70 | 692.56 | 690.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 686.00 | 691.25 | 690.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 692.40 | 690.85 | 690.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:15:00 | 692.20 | 690.85 | 690.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:15:00 | 690.50 | 690.11 | 689.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 12:15:00 | 685.50 | 689.19 | 689.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 12:15:00 | 685.50 | 689.19 | 689.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 13:15:00 | 681.20 | 687.59 | 688.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 689.20 | 687.91 | 688.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 689.20 | 687.91 | 688.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 689.20 | 687.91 | 688.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 689.20 | 687.91 | 688.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 689.00 | 688.13 | 688.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 687.50 | 688.13 | 688.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 677.00 | 685.90 | 687.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 674.55 | 683.63 | 686.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 09:30:00 | 675.00 | 677.88 | 681.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 675.25 | 677.04 | 679.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:45:00 | 672.90 | 674.88 | 678.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 667.20 | 666.09 | 670.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 668.15 | 666.09 | 670.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 667.85 | 665.88 | 668.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 654.00 | 663.12 | 666.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 640.82 | 650.32 | 654.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 641.25 | 650.32 | 654.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 641.49 | 650.32 | 654.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 639.25 | 650.32 | 654.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 11:15:00 | 621.30 | 632.75 | 641.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 626.60 | 625.75 | 634.08 | SL hit (close>ema200) qty=0.50 sl=625.75 alert=retest2 |

### Cycle 108 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 668.95 | 642.75 | 639.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 685.70 | 654.89 | 645.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 686.35 | 687.21 | 675.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 677.10 | 684.18 | 677.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 677.10 | 684.18 | 677.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 677.10 | 684.18 | 677.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 678.30 | 683.01 | 677.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 678.30 | 683.01 | 677.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 676.35 | 681.68 | 677.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 676.35 | 681.68 | 677.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 673.25 | 679.99 | 676.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 677.15 | 679.99 | 676.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 692.50 | 682.49 | 678.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 675.90 | 682.49 | 678.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 680.00 | 683.44 | 680.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:30:00 | 680.80 | 683.44 | 680.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 680.50 | 682.85 | 680.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:45:00 | 681.35 | 682.85 | 680.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 686.30 | 683.54 | 680.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 690.00 | 683.54 | 680.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:30:00 | 687.00 | 685.64 | 682.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 15:15:00 | 675.60 | 682.23 | 681.95 | SL hit (close<static) qty=1.00 sl=679.40 alert=retest2 |

### Cycle 109 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 674.40 | 680.66 | 681.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 668.10 | 674.47 | 677.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 15:15:00 | 674.45 | 668.72 | 672.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 15:15:00 | 674.45 | 668.72 | 672.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 674.45 | 668.72 | 672.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 659.60 | 668.72 | 672.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 667.00 | 668.38 | 672.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:45:00 | 657.00 | 664.79 | 669.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:15:00 | 655.30 | 664.79 | 669.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:45:00 | 656.35 | 663.69 | 668.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 654.95 | 663.69 | 668.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 624.15 | 645.69 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 622.53 | 645.69 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 623.53 | 645.69 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 622.20 | 645.69 | 654.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 635.80 | 624.63 | 636.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 635.80 | 624.63 | 636.90 | SL hit (close>ema200) qty=0.50 sl=624.63 alert=retest2 |

### Cycle 110 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 641.00 | 635.97 | 635.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 650.20 | 639.67 | 637.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 643.25 | 643.67 | 640.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 643.25 | 643.67 | 640.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 643.90 | 643.71 | 640.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:00:00 | 640.75 | 643.12 | 640.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 638.35 | 642.17 | 640.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 638.35 | 642.17 | 640.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 638.30 | 641.39 | 640.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 638.30 | 641.39 | 640.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 636.00 | 638.96 | 639.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 632.00 | 637.57 | 638.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 623.20 | 615.82 | 620.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 623.20 | 615.82 | 620.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 623.20 | 615.82 | 620.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 624.70 | 615.82 | 620.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 625.40 | 617.74 | 621.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 625.40 | 617.74 | 621.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 633.55 | 624.45 | 623.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 636.75 | 626.91 | 624.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 639.85 | 640.96 | 636.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:45:00 | 639.20 | 640.96 | 636.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 641.15 | 640.11 | 636.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 651.00 | 640.11 | 636.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 647.45 | 655.15 | 655.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 647.45 | 655.15 | 655.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 643.95 | 650.34 | 653.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 645.30 | 642.64 | 647.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 12:15:00 | 645.30 | 642.64 | 647.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 645.30 | 642.64 | 647.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:00:00 | 645.30 | 642.64 | 647.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 645.50 | 643.21 | 647.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:45:00 | 646.45 | 643.21 | 647.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 641.00 | 642.77 | 646.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:15:00 | 633.90 | 640.67 | 643.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 635.50 | 638.66 | 642.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:15:00 | 635.00 | 638.82 | 641.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 649.20 | 639.14 | 639.86 | SL hit (close>static) qty=1.00 sl=647.20 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 649.70 | 641.25 | 640.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 15:15:00 | 656.55 | 647.18 | 643.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 678.25 | 680.60 | 671.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 678.25 | 680.60 | 671.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 672.25 | 682.03 | 677.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 675.05 | 682.03 | 677.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 672.00 | 680.02 | 676.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:15:00 | 671.00 | 680.02 | 676.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 682.75 | 679.07 | 677.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 677.95 | 679.07 | 677.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 700.00 | 702.66 | 694.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 712.10 | 702.66 | 694.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 711.45 | 704.42 | 695.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 15:15:00 | 690.80 | 699.39 | 697.09 | SL hit (close<static) qty=1.00 sl=694.10 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 683.80 | 693.80 | 694.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 13:15:00 | 679.50 | 687.36 | 691.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 14:15:00 | 683.20 | 682.71 | 686.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 683.20 | 682.71 | 686.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 688.85 | 683.94 | 686.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 688.25 | 683.94 | 686.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 682.15 | 683.58 | 685.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:15:00 | 681.50 | 683.58 | 685.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:30:00 | 681.60 | 683.86 | 685.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 15:15:00 | 681.00 | 684.11 | 685.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 678.35 | 681.13 | 683.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 674.75 | 672.15 | 675.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:15:00 | 685.45 | 672.15 | 675.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 677.35 | 673.19 | 676.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 680.50 | 673.19 | 676.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 676.40 | 673.83 | 676.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 679.00 | 673.83 | 676.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 671.10 | 673.28 | 675.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 671.10 | 673.28 | 675.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 671.35 | 672.90 | 675.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 678.55 | 672.90 | 675.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 681.05 | 674.53 | 675.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 684.05 | 674.53 | 675.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 691.00 | 677.82 | 677.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 691.00 | 677.82 | 677.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 703.70 | 687.06 | 682.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 801.95 | 804.16 | 772.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 09:30:00 | 800.30 | 804.16 | 772.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 793.20 | 797.15 | 791.71 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 785.00 | 789.68 | 789.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 773.30 | 782.11 | 785.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 12:15:00 | 780.85 | 773.70 | 778.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 12:15:00 | 780.85 | 773.70 | 778.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 780.85 | 773.70 | 778.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 780.15 | 773.70 | 778.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 786.40 | 776.24 | 779.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 786.40 | 776.24 | 779.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 790.35 | 779.06 | 780.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 790.35 | 779.06 | 780.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 795.00 | 782.25 | 781.76 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 775.60 | 781.01 | 781.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 771.35 | 779.08 | 780.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 779.25 | 777.67 | 779.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 779.25 | 777.67 | 779.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 779.25 | 777.67 | 779.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 779.25 | 777.67 | 779.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 774.55 | 777.04 | 778.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:45:00 | 778.70 | 777.04 | 778.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 765.05 | 761.35 | 767.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 767.00 | 761.35 | 767.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 771.30 | 763.34 | 768.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 771.30 | 763.34 | 768.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 785.55 | 767.78 | 769.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 787.00 | 767.78 | 769.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 13:15:00 | 778.90 | 772.29 | 771.48 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 745.85 | 766.65 | 769.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 735.25 | 760.37 | 765.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 741.00 | 740.37 | 751.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:30:00 | 733.35 | 740.37 | 751.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 747.25 | 743.06 | 750.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 738.65 | 746.51 | 748.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 15:15:00 | 754.00 | 734.33 | 733.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 754.00 | 734.33 | 733.65 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 715.95 | 730.65 | 732.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 712.15 | 721.28 | 726.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 712.95 | 712.62 | 718.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:30:00 | 710.45 | 712.62 | 718.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 726.70 | 715.99 | 718.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:15:00 | 738.50 | 715.99 | 718.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 723.25 | 717.44 | 719.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 735.40 | 717.44 | 719.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 713.35 | 716.92 | 718.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 710.10 | 714.99 | 717.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 674.60 | 690.71 | 700.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 14:15:00 | 694.90 | 689.83 | 696.28 | SL hit (close>ema200) qty=0.50 sl=689.83 alert=retest2 |

### Cycle 124 — BUY (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 15:15:00 | 711.00 | 697.02 | 696.22 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 672.20 | 692.05 | 694.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 667.90 | 687.22 | 691.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 680.10 | 676.95 | 683.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:45:00 | 680.90 | 676.95 | 683.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 678.65 | 677.29 | 682.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 682.15 | 677.29 | 682.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 683.70 | 678.57 | 682.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 683.70 | 678.57 | 682.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 681.35 | 679.13 | 682.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:15:00 | 680.20 | 679.13 | 682.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 678.00 | 679.89 | 682.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 696.20 | 681.04 | 681.25 | SL hit (close>static) qty=1.00 sl=683.70 alert=retest2 |

### Cycle 126 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 690.00 | 682.83 | 682.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 704.45 | 687.16 | 684.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 739.00 | 739.34 | 722.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 739.00 | 739.34 | 722.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 719.60 | 734.05 | 723.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 719.60 | 734.05 | 723.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 719.90 | 731.22 | 722.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:30:00 | 718.45 | 731.22 | 722.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 718.60 | 728.70 | 722.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:45:00 | 717.00 | 728.70 | 722.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 723.15 | 725.23 | 722.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 718.15 | 725.23 | 722.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 717.20 | 723.63 | 721.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:30:00 | 712.15 | 723.63 | 721.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 715.90 | 722.08 | 721.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:30:00 | 714.55 | 722.08 | 721.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 709.40 | 718.68 | 719.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 706.35 | 710.41 | 712.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 698.75 | 694.28 | 698.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 698.75 | 694.28 | 698.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 698.75 | 694.28 | 698.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 698.75 | 694.28 | 698.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 699.10 | 695.25 | 698.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 697.95 | 695.25 | 698.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 696.25 | 695.45 | 698.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 694.90 | 695.16 | 698.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 660.15 | 671.31 | 682.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 667.80 | 667.28 | 675.57 | SL hit (close>ema200) qty=0.50 sl=667.28 alert=retest2 |

### Cycle 128 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 656.45 | 655.04 | 654.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 659.75 | 655.99 | 655.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 676.30 | 676.73 | 671.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 676.30 | 676.73 | 671.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 683.90 | 683.90 | 680.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 683.90 | 683.90 | 680.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 683.80 | 683.88 | 681.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 688.30 | 683.88 | 681.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 692.95 | 685.70 | 682.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:30:00 | 697.80 | 688.36 | 683.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:45:00 | 695.40 | 692.48 | 687.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 699.30 | 692.90 | 688.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-06 10:15:00 | 767.58 | 733.51 | 717.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 760.40 | 764.97 | 764.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 750.60 | 762.09 | 763.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 761.30 | 760.79 | 762.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 761.30 | 760.79 | 762.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 761.30 | 760.79 | 762.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 761.30 | 760.79 | 762.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 763.05 | 761.24 | 762.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 763.05 | 761.24 | 762.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 764.20 | 761.83 | 762.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:45:00 | 767.10 | 761.83 | 762.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 762.95 | 762.06 | 762.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 764.35 | 762.06 | 762.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 769.10 | 763.46 | 763.34 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 761.00 | 762.97 | 763.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 760.50 | 762.48 | 762.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 13:15:00 | 763.65 | 762.54 | 762.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 13:15:00 | 763.65 | 762.54 | 762.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 763.65 | 762.54 | 762.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 763.65 | 762.54 | 762.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 764.35 | 762.90 | 762.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 764.35 | 762.90 | 762.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 764.00 | 763.12 | 763.06 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 756.20 | 761.74 | 762.44 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 760.70 | 753.88 | 753.00 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 11:15:00 | 745.70 | 753.02 | 753.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 736.15 | 745.70 | 749.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 748.05 | 744.57 | 747.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 13:15:00 | 748.05 | 744.57 | 747.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 748.05 | 744.57 | 747.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 748.05 | 744.57 | 747.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 747.65 | 745.19 | 747.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 747.65 | 745.19 | 747.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 751.20 | 746.39 | 747.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 749.25 | 746.39 | 747.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 747.50 | 746.61 | 747.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 745.65 | 746.61 | 747.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 745.90 | 746.61 | 747.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:00:00 | 745.05 | 746.30 | 747.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 13:45:00 | 743.00 | 746.29 | 747.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 751.00 | 747.23 | 747.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 751.00 | 747.23 | 747.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 743.50 | 746.48 | 747.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 749.55 | 746.89 | 747.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 751.35 | 747.78 | 747.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 751.35 | 747.78 | 747.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 12:15:00 | 759.90 | 751.79 | 749.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 750.45 | 753.67 | 751.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 750.45 | 753.67 | 751.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 750.45 | 753.67 | 751.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 750.45 | 753.67 | 751.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 753.10 | 753.56 | 751.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 760.40 | 754.43 | 752.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:45:00 | 758.00 | 755.01 | 752.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 759.60 | 755.01 | 752.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:45:00 | 758.45 | 757.80 | 754.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 757.70 | 757.59 | 755.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:30:00 | 753.40 | 757.59 | 755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 758.30 | 757.44 | 755.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:45:00 | 759.05 | 757.44 | 755.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 754.60 | 756.87 | 755.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 754.60 | 756.87 | 755.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 753.00 | 756.10 | 755.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 752.40 | 754.55 | 754.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 752.40 | 754.55 | 754.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 10:15:00 | 739.10 | 751.73 | 753.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 704.80 | 703.81 | 718.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:00:00 | 704.80 | 703.81 | 718.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 678.00 | 687.67 | 693.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:45:00 | 669.85 | 680.26 | 687.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 664.00 | 679.21 | 686.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 14:15:00 | 659.45 | 658.36 | 658.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 659.45 | 658.36 | 658.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 664.00 | 659.49 | 658.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 658.60 | 659.31 | 658.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 658.60 | 659.31 | 658.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 658.60 | 659.31 | 658.82 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 651.05 | 657.66 | 658.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 11:15:00 | 650.00 | 656.13 | 657.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 652.00 | 650.28 | 652.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 652.00 | 650.28 | 652.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 654.40 | 651.10 | 653.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 654.40 | 651.10 | 653.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 656.25 | 652.13 | 653.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 655.95 | 652.13 | 653.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 652.85 | 652.56 | 653.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 648.60 | 652.56 | 653.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 650.55 | 642.77 | 643.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 652.10 | 644.64 | 644.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 652.10 | 644.64 | 644.39 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 626.85 | 645.46 | 646.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 621.80 | 640.73 | 643.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 632.10 | 631.15 | 637.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 632.10 | 631.15 | 637.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 626.00 | 625.03 | 630.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 621.00 | 624.22 | 629.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 13:15:00 | 633.45 | 627.31 | 629.01 | SL hit (close>static) qty=1.00 sl=632.20 alert=retest2 |

### Cycle 142 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 639.90 | 631.04 | 630.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 643.70 | 633.57 | 631.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 662.00 | 662.55 | 654.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 662.00 | 662.55 | 654.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 655.35 | 661.03 | 655.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 671.85 | 661.03 | 655.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 673.95 | 663.62 | 657.38 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 661.45 | 664.41 | 664.42 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 15:15:00 | 665.00 | 664.52 | 664.47 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 662.45 | 664.11 | 664.29 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 667.95 | 664.52 | 664.34 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 662.20 | 664.33 | 664.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 660.05 | 663.48 | 664.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 665.80 | 663.94 | 664.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 665.80 | 663.94 | 664.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 665.80 | 663.94 | 664.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 665.80 | 663.94 | 664.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 664.50 | 664.05 | 664.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 655.10 | 664.05 | 664.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:30:00 | 660.55 | 660.61 | 662.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 658.70 | 660.61 | 662.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:45:00 | 660.65 | 661.27 | 662.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 660.00 | 661.02 | 662.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 655.75 | 661.02 | 662.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 627.52 | 645.28 | 652.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 625.76 | 645.28 | 652.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 627.62 | 645.28 | 652.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:15:00 | 622.35 | 642.22 | 650.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:15:00 | 622.96 | 642.22 | 650.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 639.00 | 633.57 | 641.24 | SL hit (close>ema200) qty=0.50 sl=633.57 alert=retest2 |

### Cycle 148 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 649.65 | 639.59 | 638.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 660.30 | 647.13 | 642.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 689.90 | 701.30 | 686.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 689.90 | 701.30 | 686.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 689.90 | 701.30 | 686.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 689.90 | 701.30 | 686.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 687.80 | 697.49 | 687.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 687.80 | 697.49 | 687.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 670.50 | 692.09 | 685.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 670.35 | 692.09 | 685.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 675.55 | 688.78 | 684.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:30:00 | 680.70 | 685.93 | 683.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 12:15:00 | 677.70 | 681.96 | 682.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 677.70 | 681.96 | 682.41 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 685.45 | 683.09 | 682.87 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 677.25 | 682.03 | 682.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 666.00 | 677.79 | 680.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 610.00 | 608.87 | 622.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 11:00:00 | 610.00 | 608.87 | 622.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 612.65 | 607.27 | 614.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 612.65 | 607.27 | 614.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 617.30 | 610.58 | 614.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 617.30 | 610.58 | 614.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 618.35 | 612.14 | 614.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 625.10 | 612.14 | 614.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 635.55 | 616.82 | 616.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 651.95 | 634.48 | 627.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 644.95 | 645.74 | 635.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:00:00 | 644.95 | 645.74 | 635.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 632.75 | 643.14 | 634.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 632.75 | 643.14 | 634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 626.25 | 639.77 | 634.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 626.10 | 636.27 | 632.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 619.90 | 633.00 | 631.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 620.20 | 633.00 | 631.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 616.70 | 629.74 | 630.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 614.30 | 626.65 | 628.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 598.65 | 598.01 | 605.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 590.55 | 598.01 | 605.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 593.80 | 586.07 | 590.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 593.80 | 586.07 | 590.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 596.30 | 588.11 | 591.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 595.30 | 588.11 | 591.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 602.20 | 594.11 | 593.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 610.25 | 598.83 | 595.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 619.25 | 622.04 | 616.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:45:00 | 618.90 | 622.04 | 616.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 624.20 | 629.88 | 626.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 624.20 | 629.88 | 626.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 624.40 | 628.78 | 625.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 621.10 | 628.78 | 625.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 625.65 | 628.16 | 625.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:30:00 | 627.85 | 627.15 | 625.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 629.45 | 627.61 | 625.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 618.35 | 624.84 | 624.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 618.35 | 624.84 | 624.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 612.80 | 619.29 | 621.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 606.85 | 606.73 | 612.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 606.20 | 604.19 | 607.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 606.20 | 604.19 | 607.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 606.05 | 604.19 | 607.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 604.00 | 603.38 | 605.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 606.80 | 603.38 | 605.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 604.00 | 603.50 | 605.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 602.90 | 603.50 | 605.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 605.15 | 603.83 | 605.71 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 616.20 | 608.47 | 607.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 620.75 | 610.93 | 608.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 607.25 | 612.49 | 610.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 607.25 | 612.49 | 610.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 607.25 | 612.49 | 610.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 607.25 | 612.49 | 610.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 609.85 | 611.96 | 610.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 614.15 | 612.60 | 610.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 617.00 | 618.05 | 615.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 15:15:00 | 607.10 | 615.86 | 614.57 | SL hit (close<static) qty=1.00 sl=607.25 alert=retest2 |

### Cycle 157 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 581.05 | 608.90 | 611.52 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 614.25 | 610.81 | 610.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 647.90 | 624.72 | 619.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 663.45 | 664.73 | 653.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:45:00 | 663.80 | 664.73 | 653.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 661.60 | 664.75 | 659.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 660.40 | 664.75 | 659.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 665.10 | 666.31 | 662.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 660.55 | 666.31 | 662.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 663.00 | 665.65 | 662.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 671.95 | 666.50 | 663.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 15:15:00 | 674.50 | 667.37 | 664.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 13:15:00 | 739.15 | 706.55 | 687.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 723.00 | 731.26 | 731.88 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 735.10 | 732.62 | 732.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 12:15:00 | 747.20 | 736.78 | 734.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 743.35 | 743.56 | 739.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 10:30:00 | 742.75 | 743.56 | 739.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 737.85 | 742.07 | 739.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 737.85 | 742.07 | 739.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 731.15 | 739.88 | 738.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 731.15 | 739.88 | 738.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 724.00 | 736.71 | 737.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 708.50 | 728.98 | 733.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 740.80 | 721.39 | 725.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 740.80 | 721.39 | 725.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 740.80 | 721.39 | 725.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 738.70 | 721.39 | 725.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 744.35 | 725.99 | 727.25 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 744.95 | 729.78 | 728.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 746.95 | 740.65 | 735.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 762.00 | 762.00 | 754.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:45:00 | 762.75 | 762.00 | 754.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 756.75 | 760.98 | 756.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 753.70 | 760.98 | 756.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 753.95 | 759.58 | 756.47 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 15:15:00 | 749.95 | 754.85 | 755.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 741.10 | 752.10 | 753.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 747.85 | 745.75 | 748.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 747.85 | 745.75 | 748.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 747.85 | 745.75 | 748.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:30:00 | 739.00 | 745.08 | 747.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 10:45:00 | 737.40 | 741.75 | 745.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 15:15:00 | 730.00 | 723.75 | 723.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 15:15:00 | 730.00 | 723.75 | 723.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 742.00 | 728.49 | 725.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 734.00 | 735.65 | 730.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:15:00 | 731.50 | 735.65 | 730.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 732.95 | 735.11 | 730.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 730.75 | 735.11 | 730.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 729.00 | 733.89 | 730.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 729.00 | 733.89 | 730.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 730.00 | 733.11 | 730.74 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 721.35 | 728.99 | 729.16 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 743.90 | 730.71 | 729.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 10:15:00 | 756.25 | 735.82 | 732.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 11:15:00 | 751.00 | 754.36 | 746.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 12:00:00 | 751.00 | 754.36 | 746.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 743.95 | 752.28 | 746.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 743.95 | 752.28 | 746.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 743.40 | 750.50 | 745.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:45:00 | 741.95 | 750.50 | 745.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 749.90 | 750.38 | 746.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 741.00 | 750.38 | 746.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 751.00 | 750.50 | 746.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 745.65 | 750.50 | 746.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 747.25 | 749.85 | 746.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 752.00 | 749.24 | 746.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 751.80 | 750.02 | 747.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:00:00 | 753.00 | 750.65 | 748.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 739.20 | 748.18 | 747.51 | SL hit (close<static) qty=1.00 sl=742.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 736.00 | 745.74 | 746.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 733.00 | 741.63 | 744.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 734.65 | 732.42 | 735.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 734.65 | 732.42 | 735.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 734.65 | 732.42 | 735.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 729.65 | 733.56 | 735.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 743.30 | 735.78 | 735.97 | SL hit (close>static) qty=1.00 sl=738.35 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 741.70 | 735.57 | 735.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 747.90 | 739.47 | 737.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 740.20 | 740.23 | 738.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 740.35 | 740.23 | 738.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 739.15 | 739.94 | 738.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 739.15 | 739.94 | 738.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 740.00 | 739.95 | 738.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 742.00 | 739.95 | 738.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 738.85 | 745.97 | 745.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 738.85 | 745.97 | 745.99 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 746.75 | 743.58 | 743.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 753.25 | 745.26 | 744.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 746.45 | 749.03 | 746.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 746.45 | 749.03 | 746.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 746.45 | 749.03 | 746.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 745.00 | 749.03 | 746.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 749.05 | 749.03 | 746.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 749.80 | 749.49 | 747.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:00:00 | 749.25 | 767.82 | 761.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:30:00 | 750.00 | 764.24 | 760.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 735.00 | 758.39 | 758.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 735.00 | 758.39 | 758.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 730.85 | 750.10 | 754.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 703.65 | 697.90 | 707.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 703.65 | 697.90 | 707.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 703.65 | 697.90 | 707.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 707.00 | 697.90 | 707.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 704.75 | 699.27 | 706.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 705.20 | 699.27 | 706.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 715.05 | 702.23 | 705.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 716.35 | 702.23 | 705.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 711.65 | 704.12 | 705.63 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 722.35 | 707.76 | 707.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 727.20 | 716.70 | 712.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 783.65 | 786.20 | 765.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 783.65 | 786.20 | 765.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 826.05 | 837.66 | 828.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 828.05 | 837.66 | 828.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 823.85 | 834.90 | 828.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 821.80 | 834.90 | 828.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 813.20 | 823.04 | 823.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 806.00 | 819.63 | 822.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 815.90 | 813.40 | 816.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 11:15:00 | 815.90 | 813.40 | 816.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 815.90 | 813.40 | 816.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 815.90 | 813.40 | 816.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 814.45 | 813.61 | 816.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 814.50 | 813.61 | 816.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 809.00 | 812.22 | 814.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 804.65 | 810.63 | 813.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 819.60 | 813.40 | 813.89 | SL hit (close>static) qty=1.00 sl=816.15 alert=retest2 |

### Cycle 174 — BUY (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 10:15:00 | 820.50 | 814.82 | 814.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 834.10 | 818.68 | 816.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 832.25 | 836.27 | 828.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:00:00 | 832.25 | 836.27 | 828.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 841.10 | 843.40 | 838.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:45:00 | 848.70 | 843.80 | 839.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 848.30 | 845.04 | 840.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 841.15 | 851.62 | 852.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 841.15 | 851.62 | 852.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 837.00 | 848.70 | 851.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 840.00 | 837.50 | 842.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 840.00 | 837.50 | 842.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 840.00 | 837.50 | 842.28 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 848.00 | 843.43 | 843.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 853.75 | 846.14 | 844.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 849.65 | 850.38 | 848.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 849.65 | 850.38 | 848.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 849.65 | 850.38 | 848.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:15:00 | 848.25 | 850.38 | 848.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 848.25 | 849.95 | 848.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 842.00 | 849.95 | 848.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 834.30 | 846.82 | 846.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 831.50 | 837.26 | 840.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 834.35 | 834.27 | 838.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:45:00 | 834.70 | 834.27 | 838.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 833.95 | 831.22 | 834.22 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 851.60 | 838.05 | 836.75 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 833.30 | 838.86 | 839.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 825.85 | 836.25 | 838.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 825.65 | 823.38 | 829.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 825.65 | 823.38 | 829.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 830.45 | 825.25 | 829.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 832.00 | 825.25 | 829.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 828.80 | 825.96 | 829.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 829.60 | 825.96 | 829.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 825.10 | 825.78 | 828.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 825.10 | 825.78 | 828.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 837.30 | 828.09 | 829.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 839.00 | 828.09 | 829.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 845.40 | 831.55 | 830.98 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 818.15 | 831.19 | 831.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 815.45 | 825.12 | 827.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 830.50 | 824.71 | 826.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 830.50 | 824.71 | 826.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 830.50 | 824.71 | 826.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 830.50 | 824.71 | 826.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 830.00 | 825.77 | 827.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 834.20 | 825.77 | 827.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 827.15 | 826.24 | 827.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:30:00 | 830.00 | 826.24 | 827.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 828.00 | 826.59 | 827.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 828.00 | 826.59 | 827.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 828.80 | 827.04 | 827.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 828.80 | 827.04 | 827.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 830.00 | 827.63 | 827.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 830.00 | 827.63 | 827.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 15:15:00 | 835.25 | 828.74 | 828.14 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 822.40 | 827.47 | 827.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 821.05 | 826.19 | 827.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 825.65 | 822.66 | 824.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 825.65 | 822.66 | 824.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 825.65 | 822.66 | 824.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 825.65 | 822.66 | 824.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 824.80 | 823.09 | 824.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 821.60 | 823.07 | 824.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 821.00 | 822.97 | 824.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 814.65 | 820.27 | 821.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 819.95 | 821.31 | 822.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 820.60 | 821.17 | 821.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 827.55 | 822.70 | 822.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 827.55 | 822.70 | 822.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 835.00 | 827.33 | 824.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 871.70 | 876.49 | 868.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 871.70 | 876.49 | 868.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 871.70 | 876.49 | 868.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 871.30 | 876.49 | 868.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 875.00 | 876.19 | 869.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 874.20 | 876.19 | 869.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 861.35 | 872.70 | 868.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 860.30 | 872.70 | 868.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 861.65 | 870.49 | 868.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 871.60 | 870.49 | 868.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 873.10 | 878.50 | 878.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 873.10 | 878.50 | 878.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 865.80 | 875.96 | 877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 854.05 | 850.67 | 859.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 857.85 | 850.67 | 859.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 856.30 | 851.80 | 859.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 848.85 | 856.53 | 858.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 806.41 | 820.56 | 831.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 826.85 | 816.59 | 823.52 | SL hit (close>ema200) qty=0.50 sl=816.59 alert=retest2 |

### Cycle 186 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 833.05 | 827.14 | 827.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 834.35 | 828.59 | 827.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 864.15 | 865.03 | 851.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:15:00 | 863.80 | 865.03 | 851.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 849.05 | 858.48 | 853.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 849.05 | 858.48 | 853.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 848.85 | 856.56 | 853.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 849.60 | 856.56 | 853.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 855.15 | 856.03 | 853.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 852.60 | 856.03 | 853.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 860.20 | 857.01 | 854.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 858.70 | 857.01 | 854.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 853.25 | 856.77 | 854.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:15:00 | 851.55 | 856.77 | 854.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 854.65 | 856.35 | 854.83 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 846.70 | 852.74 | 853.42 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 863.60 | 853.83 | 853.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 884.25 | 863.13 | 858.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 880.00 | 880.16 | 871.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 14:00:00 | 884.20 | 880.97 | 872.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 14:45:00 | 885.25 | 881.85 | 873.82 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 887.50 | 906.53 | 899.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 887.50 | 906.53 | 899.13 | SL hit (close<ema400) qty=1.00 sl=899.13 alert=retest1 |

### Cycle 189 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 885.45 | 895.26 | 895.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 873.20 | 890.84 | 893.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 890.00 | 888.08 | 890.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 12:15:00 | 890.00 | 888.08 | 890.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 890.00 | 888.08 | 890.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 890.00 | 888.08 | 890.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 897.00 | 888.84 | 890.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 898.30 | 888.84 | 890.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 894.45 | 889.96 | 890.55 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 896.70 | 891.31 | 891.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 908.30 | 894.71 | 892.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 14:15:00 | 913.05 | 917.51 | 908.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 913.05 | 917.51 | 908.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 909.50 | 915.90 | 908.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 911.95 | 915.90 | 908.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 927.20 | 918.16 | 910.64 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 896.75 | 910.17 | 911.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 888.50 | 905.83 | 909.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 906.10 | 902.13 | 906.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 906.10 | 902.13 | 906.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 906.10 | 902.13 | 906.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:45:00 | 907.30 | 902.13 | 906.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 897.00 | 901.11 | 905.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 888.55 | 901.11 | 905.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:15:00 | 844.12 | 866.23 | 881.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 880.00 | 863.63 | 874.82 | SL hit (close>ema200) qty=0.50 sl=863.63 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 893.80 | 875.80 | 875.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 895.20 | 887.06 | 881.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 882.40 | 886.16 | 882.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 882.40 | 886.16 | 882.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 882.40 | 886.16 | 882.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 882.40 | 886.16 | 882.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 881.50 | 885.23 | 882.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:15:00 | 881.00 | 885.23 | 882.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 878.40 | 883.86 | 881.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 878.40 | 883.86 | 881.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 878.85 | 881.31 | 880.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 880.05 | 881.31 | 880.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 882.15 | 881.48 | 881.07 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 877.50 | 880.68 | 880.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 875.30 | 878.78 | 879.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 871.35 | 868.81 | 871.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 871.35 | 868.81 | 871.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 871.35 | 868.81 | 871.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 871.35 | 868.81 | 871.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 866.50 | 868.35 | 871.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 865.50 | 868.35 | 871.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 865.45 | 867.77 | 870.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 882.80 | 870.37 | 871.23 | SL hit (close>static) qty=1.00 sl=872.75 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 895.00 | 875.30 | 873.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 905.60 | 892.07 | 884.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 893.85 | 895.86 | 889.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 893.85 | 895.86 | 889.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 885.85 | 895.04 | 892.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 885.85 | 895.04 | 892.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 883.80 | 892.79 | 891.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 883.50 | 892.79 | 891.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 894.00 | 892.03 | 891.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 887.45 | 892.03 | 891.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 886.00 | 890.82 | 890.90 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 900.00 | 892.66 | 891.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 903.85 | 894.90 | 892.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 918.80 | 921.84 | 909.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 918.80 | 921.84 | 909.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 927.90 | 920.50 | 911.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 934.95 | 922.83 | 913.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:30:00 | 934.00 | 927.68 | 917.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 933.05 | 932.81 | 923.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:00:00 | 933.95 | 933.71 | 925.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 927.35 | 931.91 | 926.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 928.00 | 931.91 | 926.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 925.75 | 930.68 | 926.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 925.75 | 930.68 | 926.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 926.85 | 929.91 | 926.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 938.00 | 932.18 | 927.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:45:00 | 935.55 | 933.95 | 929.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 938.80 | 934.35 | 932.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 934.50 | 934.35 | 932.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 938.85 | 935.25 | 932.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 940.00 | 936.20 | 933.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 940.25 | 937.30 | 934.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 946.35 | 942.77 | 937.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 934.85 | 941.51 | 941.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 934.85 | 941.51 | 941.60 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 945.80 | 941.46 | 941.44 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 933.00 | 941.60 | 941.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 930.30 | 938.62 | 939.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 934.80 | 931.56 | 935.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:00:00 | 934.80 | 931.56 | 935.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 930.30 | 931.31 | 934.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 934.00 | 931.31 | 934.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 920.45 | 929.13 | 933.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 929.20 | 929.13 | 933.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 928.15 | 927.44 | 931.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 928.15 | 927.44 | 931.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 928.00 | 927.55 | 931.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 929.30 | 927.55 | 931.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 952.00 | 932.44 | 933.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 952.00 | 932.44 | 933.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 947.40 | 935.43 | 934.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 958.35 | 946.01 | 940.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 899.60 | 938.96 | 938.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 899.60 | 938.96 | 938.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 899.60 | 938.96 | 938.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:15:00 | 882.85 | 938.96 | 938.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 885.05 | 928.18 | 933.33 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 937.95 | 926.38 | 926.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 946.00 | 933.77 | 930.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 963.70 | 972.24 | 966.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 15:15:00 | 963.70 | 972.24 | 966.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 963.70 | 972.24 | 966.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 983.00 | 969.78 | 967.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 981.65 | 990.67 | 991.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 981.65 | 990.67 | 991.75 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 1024.00 | 997.34 | 994.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 10:15:00 | 1069.95 | 1011.86 | 1001.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 10:15:00 | 1030.85 | 1034.59 | 1021.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:00:00 | 1030.85 | 1034.59 | 1021.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1027.75 | 1031.42 | 1024.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1008.85 | 1031.42 | 1024.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1019.85 | 1029.11 | 1024.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:00:00 | 1027.80 | 1028.17 | 1024.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 15:00:00 | 1029.90 | 1027.91 | 1025.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 1028.00 | 1027.83 | 1025.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 13:15:00 | 1020.50 | 1024.49 | 1024.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 1020.50 | 1024.49 | 1024.58 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1029.65 | 1025.49 | 1025.01 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 1023.95 | 1024.67 | 1024.74 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 1054.90 | 1029.74 | 1026.90 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 1035.50 | 1046.43 | 1047.47 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 1055.00 | 1047.64 | 1046.77 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 13:15:00 | 1023.50 | 1044.88 | 1046.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 998.70 | 1026.74 | 1034.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1017.70 | 1013.48 | 1022.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:45:00 | 1014.80 | 1013.48 | 1022.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1010.00 | 998.58 | 1004.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 14:45:00 | 995.00 | 1004.38 | 1005.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1009.10 | 1006.74 | 1006.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 1009.10 | 1006.74 | 1006.52 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 991.50 | 1004.85 | 1005.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 963.00 | 987.88 | 994.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 984.60 | 977.47 | 985.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 984.60 | 977.47 | 985.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 984.00 | 978.78 | 985.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 988.80 | 978.78 | 985.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1007.40 | 984.50 | 987.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1007.40 | 984.50 | 987.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 994.20 | 986.44 | 988.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 991.50 | 986.44 | 988.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 992.40 | 986.67 | 987.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1006.10 | 990.56 | 989.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1006.10 | 990.56 | 989.34 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 982.50 | 994.59 | 995.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 971.40 | 983.27 | 988.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 977.40 | 975.74 | 982.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 977.40 | 975.74 | 982.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 982.20 | 977.03 | 982.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 982.20 | 977.03 | 982.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 985.30 | 978.68 | 982.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 985.30 | 978.68 | 982.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 989.90 | 980.93 | 983.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 989.90 | 980.93 | 983.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 999.80 | 984.70 | 984.68 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 953.00 | 981.49 | 983.54 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 1003.20 | 981.51 | 978.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 1037.90 | 996.22 | 985.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 12:15:00 | 1003.20 | 1006.04 | 995.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1003.20 | 1006.04 | 995.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 997.50 | 1005.34 | 998.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 999.00 | 1005.34 | 998.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 997.20 | 1003.71 | 998.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 996.00 | 1003.71 | 998.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 997.80 | 1002.53 | 998.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:45:00 | 999.30 | 1002.02 | 998.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:45:00 | 999.60 | 1001.42 | 998.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 999.00 | 999.17 | 998.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 991.00 | 997.53 | 997.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 991.00 | 997.53 | 997.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 982.50 | 991.76 | 994.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 996.60 | 977.67 | 981.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 996.60 | 977.67 | 981.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 996.60 | 977.67 | 981.19 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 997.20 | 984.65 | 983.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 1006.00 | 992.95 | 988.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 991.90 | 993.74 | 989.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 991.90 | 993.74 | 989.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 991.90 | 993.74 | 989.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 991.20 | 993.74 | 989.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 986.00 | 992.19 | 989.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 986.00 | 992.19 | 989.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 972.80 | 988.31 | 987.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 972.80 | 988.31 | 987.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 981.30 | 986.91 | 987.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 968.90 | 979.53 | 983.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 936.70 | 930.67 | 944.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:45:00 | 936.00 | 930.67 | 944.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 938.10 | 932.58 | 942.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 942.30 | 932.58 | 942.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 959.10 | 937.88 | 943.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 959.10 | 937.88 | 943.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 955.00 | 941.31 | 944.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 945.00 | 941.31 | 944.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 943.00 | 940.98 | 942.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 946.60 | 943.30 | 943.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 946.60 | 943.30 | 943.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 13:15:00 | 950.30 | 944.70 | 943.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 942.20 | 944.49 | 943.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 942.20 | 944.49 | 943.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 942.20 | 944.49 | 943.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 922.00 | 944.49 | 943.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 923.20 | 940.23 | 941.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 916.00 | 931.86 | 937.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 930.90 | 926.54 | 933.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 930.90 | 926.54 | 933.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 935.10 | 928.25 | 933.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 921.10 | 928.25 | 933.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 915.50 | 925.70 | 931.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 913.20 | 920.81 | 927.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:30:00 | 914.30 | 916.31 | 924.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 943.00 | 922.10 | 920.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 943.00 | 922.10 | 920.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 960.50 | 936.80 | 928.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 942.30 | 949.30 | 940.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 942.30 | 949.30 | 940.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 942.90 | 948.02 | 940.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 921.40 | 948.02 | 940.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 941.50 | 946.72 | 941.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 950.40 | 945.63 | 941.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 949.60 | 944.76 | 941.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 961.90 | 984.60 | 987.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 961.90 | 984.60 | 987.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 950.90 | 967.92 | 973.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 974.80 | 966.72 | 971.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 974.80 | 966.72 | 971.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 974.80 | 966.72 | 971.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 974.80 | 966.72 | 971.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 975.70 | 968.51 | 971.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 975.00 | 968.51 | 971.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 971.00 | 970.37 | 971.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 975.30 | 970.37 | 971.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 975.90 | 971.47 | 972.25 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 978.30 | 972.84 | 972.80 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 972.40 | 973.26 | 973.33 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 14:15:00 | 978.90 | 974.24 | 973.73 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 972.20 | 974.74 | 974.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 966.10 | 972.34 | 973.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 972.90 | 972.45 | 973.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 972.90 | 972.45 | 973.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 972.90 | 972.45 | 973.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 972.90 | 972.45 | 973.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 973.80 | 972.72 | 973.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 953.20 | 972.72 | 973.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 905.54 | 924.82 | 930.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 11:15:00 | 857.88 | 890.61 | 910.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 854.10 | 834.80 | 833.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 860.00 | 839.84 | 835.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 864.25 | 864.82 | 853.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 864.25 | 864.82 | 853.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 842.10 | 859.31 | 854.02 | EMA400 retest candle locked (from upside) |

### Cycle 231 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 836.90 | 851.68 | 852.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 831.25 | 845.85 | 849.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 10:15:00 | 838.25 | 836.38 | 841.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 838.25 | 836.38 | 841.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 838.25 | 836.38 | 841.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:00:00 | 838.25 | 836.38 | 841.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 841.35 | 837.37 | 841.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:45:00 | 834.60 | 837.27 | 841.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 836.70 | 837.41 | 841.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 837.10 | 838.34 | 840.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 847.70 | 840.22 | 841.52 | SL hit (close>static) qty=1.00 sl=844.90 alert=retest2 |

### Cycle 232 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 914.00 | 854.78 | 847.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 946.30 | 873.08 | 856.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 864.55 | 879.98 | 865.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 864.55 | 879.98 | 865.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 864.55 | 879.98 | 865.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 864.55 | 879.98 | 865.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 857.20 | 875.42 | 865.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 846.35 | 875.42 | 865.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 853.00 | 870.94 | 864.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:15:00 | 863.60 | 863.45 | 861.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 845.40 | 859.98 | 860.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 845.40 | 859.98 | 860.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 837.50 | 855.49 | 858.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 11:15:00 | 817.75 | 817.44 | 831.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 11:30:00 | 824.45 | 817.44 | 831.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 816.00 | 798.51 | 809.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 816.00 | 798.51 | 809.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 817.50 | 802.31 | 810.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 825.00 | 802.31 | 810.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 840.75 | 816.77 | 815.74 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 810.45 | 820.51 | 820.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 807.20 | 813.62 | 817.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 822.25 | 803.34 | 809.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 822.25 | 803.34 | 809.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 822.25 | 803.34 | 809.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 822.25 | 803.34 | 809.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 816.40 | 805.95 | 809.75 | EMA400 retest candle locked (from downside) |

### Cycle 236 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 823.60 | 812.03 | 812.02 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 797.95 | 810.27 | 811.72 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 822.00 | 812.63 | 812.23 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 803.55 | 812.83 | 813.13 | EMA200 below EMA400 |

### Cycle 240 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 855.00 | 818.44 | 815.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 857.65 | 826.28 | 818.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 854.70 | 855.50 | 845.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 865.50 | 855.50 | 845.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 849.60 | 864.62 | 857.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 849.60 | 864.62 | 857.52 | SL hit (close<ema400) qty=1.00 sl=857.52 alert=retest1 |

### Cycle 241 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 829.50 | 852.37 | 852.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 829.00 | 847.70 | 850.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 845.25 | 838.91 | 844.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 845.25 | 838.91 | 844.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 845.25 | 838.91 | 844.75 | EMA400 retest candle locked (from downside) |

### Cycle 242 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 855.30 | 845.98 | 845.27 | EMA200 above EMA400 |

### Cycle 243 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 835.75 | 844.01 | 845.13 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 857.85 | 847.61 | 846.62 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 841.50 | 846.42 | 846.97 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 858.05 | 848.74 | 847.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 860.75 | 851.15 | 849.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 11:15:00 | 860.65 | 860.91 | 857.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 11:30:00 | 861.25 | 860.91 | 857.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 859.15 | 860.55 | 858.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 859.15 | 860.55 | 858.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 855.40 | 859.52 | 858.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 854.55 | 859.52 | 858.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 855.15 | 858.65 | 857.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 850.50 | 858.65 | 857.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 247 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 848.35 | 856.59 | 856.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 838.10 | 852.89 | 855.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 851.60 | 843.13 | 847.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 851.60 | 843.13 | 847.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 851.60 | 843.13 | 847.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 851.60 | 843.13 | 847.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 856.35 | 845.78 | 848.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 855.20 | 845.78 | 848.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 248 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 856.35 | 850.84 | 850.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 862.70 | 853.91 | 852.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 860.60 | 865.06 | 861.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 860.60 | 865.06 | 861.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 860.60 | 865.06 | 861.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 860.60 | 865.06 | 861.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 851.35 | 862.32 | 860.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 851.35 | 862.32 | 860.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 857.90 | 861.44 | 860.29 | EMA400 retest candle locked (from upside) |

### Cycle 249 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 835.95 | 856.34 | 858.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 830.30 | 847.60 | 853.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 840.90 | 839.91 | 846.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 840.90 | 839.91 | 846.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 840.90 | 839.91 | 846.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 832.45 | 839.91 | 846.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 836.30 | 838.02 | 841.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:45:00 | 839.65 | 838.31 | 840.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 838.00 | 839.21 | 840.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 838.00 | 838.97 | 840.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 847.65 | 838.97 | 840.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 839.00 | 838.97 | 840.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 833.60 | 837.90 | 839.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 845.95 | 840.56 | 840.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 250 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 845.95 | 840.56 | 840.15 | EMA200 above EMA400 |

### Cycle 251 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 837.60 | 841.82 | 841.98 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 09:15:00 | 502.85 | 2023-05-15 14:15:00 | 483.50 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2023-05-26 11:30:00 | 459.20 | 2023-05-29 11:15:00 | 456.65 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-05-26 14:00:00 | 458.60 | 2023-05-29 11:15:00 | 456.65 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-05-26 15:15:00 | 459.00 | 2023-05-29 11:15:00 | 456.65 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-05-29 11:15:00 | 458.45 | 2023-05-29 11:15:00 | 456.65 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-06-16 09:15:00 | 478.40 | 2023-06-22 10:15:00 | 488.25 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest1 | 2023-07-05 09:15:00 | 515.05 | 2023-07-05 13:15:00 | 540.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-07-05 10:15:00 | 516.05 | 2023-07-05 13:15:00 | 541.85 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-07-05 09:15:00 | 515.05 | 2023-07-06 12:15:00 | 529.30 | STOP_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2023-07-05 10:15:00 | 516.05 | 2023-07-06 12:15:00 | 529.30 | STOP_HIT | 0.50 | 2.57% |
| BUY | retest2 | 2023-07-07 13:30:00 | 536.15 | 2023-07-11 15:15:00 | 525.05 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-07-10 09:30:00 | 539.10 | 2023-07-11 15:15:00 | 525.05 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2023-07-10 11:45:00 | 535.80 | 2023-07-11 15:15:00 | 525.05 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-07-10 12:30:00 | 534.50 | 2023-07-11 15:15:00 | 525.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2023-07-26 13:45:00 | 568.25 | 2023-07-27 11:15:00 | 555.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-08-04 13:30:00 | 535.25 | 2023-08-10 10:15:00 | 533.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-08-07 10:15:00 | 534.45 | 2023-08-10 10:15:00 | 533.90 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2023-09-27 12:15:00 | 609.20 | 2023-09-28 11:15:00 | 616.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-09-27 15:00:00 | 607.95 | 2023-09-28 11:15:00 | 616.45 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-10-09 09:15:00 | 595.45 | 2023-10-18 15:15:00 | 565.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-09 09:15:00 | 595.45 | 2023-10-19 11:15:00 | 582.65 | STOP_HIT | 0.50 | 2.15% |
| BUY | retest2 | 2023-11-01 10:00:00 | 563.90 | 2023-11-01 12:15:00 | 554.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-11-06 11:15:00 | 552.55 | 2023-11-08 10:15:00 | 559.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-11-07 09:15:00 | 552.90 | 2023-11-08 10:15:00 | 559.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-11-16 09:15:00 | 575.35 | 2023-11-16 12:15:00 | 570.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-11-16 11:30:00 | 575.80 | 2023-11-16 12:15:00 | 570.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-11-24 09:30:00 | 582.40 | 2023-11-24 12:15:00 | 568.05 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2023-12-04 09:15:00 | 574.40 | 2023-12-06 10:15:00 | 569.45 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-12-04 10:15:00 | 577.80 | 2023-12-06 10:15:00 | 569.45 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest1 | 2023-12-19 09:15:00 | 570.95 | 2023-12-19 12:15:00 | 562.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-12-20 09:15:00 | 567.40 | 2023-12-20 13:15:00 | 551.40 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2023-12-20 11:00:00 | 567.10 | 2023-12-20 13:15:00 | 551.40 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-01-02 09:15:00 | 573.00 | 2024-01-02 11:15:00 | 564.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-01-19 11:00:00 | 554.15 | 2024-01-23 09:15:00 | 596.65 | STOP_HIT | 1.00 | -7.67% |
| SELL | retest2 | 2024-01-19 11:30:00 | 555.00 | 2024-01-23 09:15:00 | 596.65 | STOP_HIT | 1.00 | -7.50% |
| SELL | retest2 | 2024-02-01 10:15:00 | 527.90 | 2024-02-05 14:15:00 | 534.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-02-01 13:45:00 | 527.55 | 2024-02-05 14:15:00 | 534.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-02-02 13:00:00 | 526.40 | 2024-02-05 14:15:00 | 534.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2024-02-08 09:15:00 | 553.00 | 2024-02-08 14:15:00 | 538.40 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-02-16 09:15:00 | 545.50 | 2024-02-19 14:15:00 | 538.35 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-02-19 10:15:00 | 542.80 | 2024-02-19 14:15:00 | 538.35 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-02-19 11:00:00 | 542.90 | 2024-02-19 14:15:00 | 538.35 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-02-23 10:15:00 | 534.65 | 2024-02-23 12:15:00 | 537.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-02-23 15:15:00 | 535.00 | 2024-02-26 13:15:00 | 537.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-02-26 10:00:00 | 534.55 | 2024-02-26 13:15:00 | 537.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-03-06 11:30:00 | 526.60 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2024-03-06 13:00:00 | 525.60 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2024-03-06 14:15:00 | 526.30 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-03-07 12:15:00 | 527.05 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2024-03-07 15:00:00 | 520.40 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-03-11 14:45:00 | 523.95 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-03-14 11:15:00 | 521.80 | 2024-03-14 13:15:00 | 519.75 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-03-22 10:15:00 | 526.55 | 2024-04-05 09:15:00 | 579.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-22 11:15:00 | 526.05 | 2024-04-05 09:15:00 | 578.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-22 13:15:00 | 526.75 | 2024-04-05 09:15:00 | 579.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-18 10:30:00 | 583.20 | 2024-04-23 11:15:00 | 588.95 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-04-18 11:30:00 | 582.55 | 2024-04-23 11:15:00 | 588.95 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-04-18 13:00:00 | 583.50 | 2024-04-23 11:15:00 | 588.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-04-26 11:30:00 | 605.95 | 2024-05-02 09:15:00 | 600.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-04-26 13:45:00 | 605.55 | 2024-05-02 09:15:00 | 600.30 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-04-29 14:45:00 | 603.80 | 2024-05-02 09:15:00 | 600.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-04-30 09:15:00 | 604.10 | 2024-05-02 09:15:00 | 600.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-06-13 12:15:00 | 616.05 | 2024-06-19 09:15:00 | 677.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 15:15:00 | 622.70 | 2024-06-19 09:15:00 | 684.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 09:30:00 | 692.40 | 2024-07-05 12:15:00 | 685.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-05 10:15:00 | 692.20 | 2024-07-05 12:15:00 | 685.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-05 12:15:00 | 690.50 | 2024-07-05 12:15:00 | 685.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-07-08 11:00:00 | 674.55 | 2024-07-18 09:15:00 | 640.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 09:30:00 | 675.00 | 2024-07-18 09:15:00 | 641.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 14:30:00 | 675.25 | 2024-07-18 09:15:00 | 641.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 09:45:00 | 672.90 | 2024-07-18 09:15:00 | 639.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 10:45:00 | 654.00 | 2024-07-19 11:15:00 | 621.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 11:00:00 | 674.55 | 2024-07-22 09:15:00 | 626.60 | STOP_HIT | 0.50 | 7.11% |
| SELL | retest2 | 2024-07-09 09:30:00 | 675.00 | 2024-07-22 09:15:00 | 626.60 | STOP_HIT | 0.50 | 7.17% |
| SELL | retest2 | 2024-07-09 14:30:00 | 675.25 | 2024-07-22 09:15:00 | 626.60 | STOP_HIT | 0.50 | 7.20% |
| SELL | retest2 | 2024-07-10 09:45:00 | 672.90 | 2024-07-22 09:15:00 | 626.60 | STOP_HIT | 0.50 | 6.88% |
| SELL | retest2 | 2024-07-12 10:45:00 | 654.00 | 2024-07-22 09:15:00 | 626.60 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2024-07-29 09:15:00 | 690.00 | 2024-07-29 15:15:00 | 675.60 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-29 11:30:00 | 687.00 | 2024-07-29 15:15:00 | 675.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-08-01 13:45:00 | 657.00 | 2024-08-05 09:15:00 | 624.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 14:15:00 | 655.30 | 2024-08-05 09:15:00 | 622.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 14:45:00 | 656.35 | 2024-08-05 09:15:00 | 623.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 15:15:00 | 654.95 | 2024-08-05 09:15:00 | 622.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 13:45:00 | 657.00 | 2024-08-06 09:15:00 | 635.80 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2024-08-01 14:15:00 | 655.30 | 2024-08-06 09:15:00 | 635.80 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-08-01 14:45:00 | 656.35 | 2024-08-06 09:15:00 | 635.80 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-08-01 15:15:00 | 654.95 | 2024-08-06 09:15:00 | 635.80 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2024-08-06 14:00:00 | 627.65 | 2024-08-07 13:15:00 | 638.55 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-06 14:30:00 | 627.55 | 2024-08-07 13:15:00 | 638.55 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-08-07 09:15:00 | 627.30 | 2024-08-07 13:15:00 | 638.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-07 11:00:00 | 627.70 | 2024-08-07 13:15:00 | 638.55 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-08-20 15:15:00 | 651.00 | 2024-08-28 09:15:00 | 647.45 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-08-30 15:15:00 | 633.90 | 2024-09-03 10:15:00 | 649.20 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-09-02 09:30:00 | 635.50 | 2024-09-03 10:15:00 | 649.20 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-09-02 12:15:00 | 635.00 | 2024-09-03 10:15:00 | 649.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-09-11 09:15:00 | 712.10 | 2024-09-11 15:15:00 | 690.80 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-09-11 10:00:00 | 711.45 | 2024-09-11 15:15:00 | 690.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-09-16 10:15:00 | 681.50 | 2024-09-19 10:15:00 | 691.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-09-16 13:30:00 | 681.60 | 2024-09-19 10:15:00 | 691.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-09-16 15:15:00 | 681.00 | 2024-09-19 10:15:00 | 691.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-09-17 09:30:00 | 678.35 | 2024-09-19 10:15:00 | 691.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-10-11 09:15:00 | 738.65 | 2024-10-16 15:15:00 | 754.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-21 14:45:00 | 710.10 | 2024-10-23 09:15:00 | 674.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 710.10 | 2024-10-23 14:15:00 | 694.90 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2024-10-28 14:15:00 | 680.20 | 2024-10-29 14:15:00 | 696.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-10-28 15:15:00 | 678.00 | 2024-10-29 14:15:00 | 696.20 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-11-12 12:45:00 | 694.90 | 2024-11-13 14:15:00 | 660.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 694.90 | 2024-11-14 12:15:00 | 667.80 | STOP_HIT | 0.50 | 3.90% |
| BUY | retest2 | 2024-12-03 10:30:00 | 697.80 | 2024-12-06 10:15:00 | 767.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 14:45:00 | 695.40 | 2024-12-06 10:15:00 | 764.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-04 09:15:00 | 699.30 | 2024-12-06 10:15:00 | 769.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-27 10:45:00 | 745.65 | 2024-12-30 10:15:00 | 751.35 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-27 11:15:00 | 745.90 | 2024-12-30 10:15:00 | 751.35 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-12-27 12:00:00 | 745.05 | 2024-12-30 10:15:00 | 751.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-27 13:45:00 | 743.00 | 2024-12-30 10:15:00 | 751.35 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-12-31 13:15:00 | 760.40 | 2025-01-02 14:15:00 | 752.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-12-31 13:45:00 | 758.00 | 2025-01-02 14:15:00 | 752.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-12-31 14:15:00 | 759.60 | 2025-01-02 14:15:00 | 752.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-01 09:45:00 | 758.45 | 2025-01-02 14:15:00 | 752.40 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-01-10 14:45:00 | 669.85 | 2025-01-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-01-13 09:15:00 | 664.00 | 2025-01-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-01-21 10:15:00 | 648.60 | 2025-01-23 14:15:00 | 652.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-01-23 14:15:00 | 650.55 | 2025-01-23 14:15:00 | 652.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-01-28 15:00:00 | 621.00 | 2025-01-29 13:15:00 | 633.45 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 655.10 | 2025-02-12 09:15:00 | 627.52 | PARTIAL | 0.50 | 4.21% |
| SELL | retest2 | 2025-02-10 13:30:00 | 660.55 | 2025-02-12 09:15:00 | 625.76 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-02-10 14:00:00 | 658.70 | 2025-02-12 09:15:00 | 627.62 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-02-10 14:45:00 | 660.65 | 2025-02-12 10:15:00 | 622.35 | PARTIAL | 0.50 | 5.80% |
| SELL | retest2 | 2025-02-11 09:15:00 | 655.75 | 2025-02-12 10:15:00 | 622.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 655.10 | 2025-02-13 09:15:00 | 639.00 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-02-10 13:30:00 | 660.55 | 2025-02-13 09:15:00 | 639.00 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2025-02-10 14:00:00 | 658.70 | 2025-02-13 09:15:00 | 639.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-02-10 14:45:00 | 660.65 | 2025-02-13 09:15:00 | 639.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-02-11 09:15:00 | 655.75 | 2025-02-13 09:15:00 | 639.00 | STOP_HIT | 0.50 | 2.55% |
| BUY | retest2 | 2025-02-24 09:30:00 | 680.70 | 2025-02-24 12:15:00 | 677.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-03-25 13:30:00 | 627.85 | 2025-03-26 09:15:00 | 618.35 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-03-25 15:00:00 | 629.45 | 2025-03-26 09:15:00 | 618.35 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-04-03 11:30:00 | 614.15 | 2025-04-04 15:15:00 | 607.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-04-04 15:00:00 | 617.00 | 2025-04-04 15:15:00 | 607.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-04-21 13:00:00 | 671.95 | 2025-04-22 13:15:00 | 739.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 15:15:00 | 674.50 | 2025-04-22 13:15:00 | 741.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-20 13:30:00 | 739.00 | 2025-05-26 15:15:00 | 730.00 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-05-21 10:45:00 | 737.40 | 2025-05-26 15:15:00 | 730.00 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-06-02 12:15:00 | 752.00 | 2025-06-03 09:15:00 | 739.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-02 12:45:00 | 751.80 | 2025-06-03 09:15:00 | 739.20 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-06-02 15:00:00 | 753.00 | 2025-06-03 09:15:00 | 739.20 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-05 14:30:00 | 729.65 | 2025-06-06 09:15:00 | 743.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-06-06 10:45:00 | 728.40 | 2025-06-06 14:15:00 | 741.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-06 13:00:00 | 729.80 | 2025-06-06 14:15:00 | 741.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-11 09:15:00 | 742.00 | 2025-06-13 09:15:00 | 738.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-17 14:30:00 | 749.80 | 2025-06-19 14:15:00 | 735.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-06-19 13:00:00 | 749.25 | 2025-06-19 14:15:00 | 735.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-06-19 13:30:00 | 750.00 | 2025-06-19 14:15:00 | 735.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-07-10 10:45:00 | 804.65 | 2025-07-11 09:15:00 | 819.60 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-07-16 11:45:00 | 848.70 | 2025-07-18 15:15:00 | 841.15 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-16 12:45:00 | 848.30 | 2025-07-18 15:15:00 | 841.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-12 12:15:00 | 821.60 | 2025-08-14 11:15:00 | 827.55 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-12 13:15:00 | 821.00 | 2025-08-14 11:15:00 | 827.55 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-08-13 13:45:00 | 814.65 | 2025-08-14 11:15:00 | 827.55 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-08-14 09:15:00 | 819.95 | 2025-08-14 11:15:00 | 827.55 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-22 09:15:00 | 871.60 | 2025-08-26 15:15:00 | 873.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-09-02 12:00:00 | 848.85 | 2025-09-05 09:15:00 | 806.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-02 12:00:00 | 848.85 | 2025-09-08 09:15:00 | 826.85 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest1 | 2025-09-16 14:00:00 | 884.20 | 2025-09-19 09:15:00 | 887.50 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest1 | 2025-09-16 14:45:00 | 885.25 | 2025-09-19 09:15:00 | 887.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-29 09:15:00 | 888.55 | 2025-09-30 11:15:00 | 844.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 09:15:00 | 888.55 | 2025-09-30 15:15:00 | 880.00 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2025-10-09 14:15:00 | 865.50 | 2025-10-10 10:15:00 | 882.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-09 15:00:00 | 865.45 | 2025-10-10 10:15:00 | 882.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-17 10:30:00 | 934.95 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-10-17 12:30:00 | 934.00 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-10-20 10:15:00 | 933.05 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-10-20 12:00:00 | 933.95 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-10-21 13:45:00 | 938.00 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-10-23 10:45:00 | 935.55 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-10-24 09:45:00 | 938.80 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-24 10:15:00 | 934.50 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-10-24 11:45:00 | 940.00 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-24 14:00:00 | 940.25 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-27 09:45:00 | 946.35 | 2025-10-28 14:15:00 | 934.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-17 09:15:00 | 983.00 | 2025-11-21 15:15:00 | 981.65 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-11-26 12:00:00 | 1027.80 | 2025-11-27 13:15:00 | 1020.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-26 15:00:00 | 1029.90 | 2025-11-27 13:15:00 | 1020.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-27 10:00:00 | 1028.00 | 2025-11-27 13:15:00 | 1020.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-12 14:45:00 | 995.00 | 2025-12-15 11:15:00 | 1009.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-19 11:15:00 | 991.50 | 2025-12-19 14:15:00 | 1006.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-19 13:45:00 | 992.40 | 2025-12-19 14:15:00 | 1006.10 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-01 12:45:00 | 999.30 | 2026-01-02 10:15:00 | 991.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-01 13:45:00 | 999.60 | 2026-01-02 10:15:00 | 991.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-02 09:45:00 | 999.00 | 2026-01-02 10:15:00 | 991.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-14 09:15:00 | 945.00 | 2026-01-16 12:15:00 | 946.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-01-14 15:15:00 | 943.00 | 2026-01-16 12:15:00 | 946.60 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-20 13:00:00 | 913.20 | 2026-01-22 09:15:00 | 943.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-01-20 14:30:00 | 914.30 | 2026-01-22 09:15:00 | 943.00 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-01-27 11:30:00 | 950.40 | 2026-02-04 10:15:00 | 961.90 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2026-01-27 14:15:00 | 949.60 | 2026-02-04 10:15:00 | 961.90 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 953.20 | 2026-02-27 14:15:00 | 905.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 953.20 | 2026-03-02 11:15:00 | 857.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-16 12:45:00 | 834.60 | 2026-03-17 09:15:00 | 847.70 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-03-16 14:15:00 | 836.70 | 2026-03-17 09:15:00 | 847.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-17 09:15:00 | 837.10 | 2026-03-17 09:15:00 | 847.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-03-17 11:15:00 | 835.15 | 2026-03-18 09:15:00 | 914.00 | STOP_HIT | 1.00 | -9.44% |
| BUY | retest2 | 2026-03-19 13:15:00 | 863.60 | 2026-03-19 14:15:00 | 845.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2026-04-10 09:15:00 | 865.50 | 2026-04-13 09:15:00 | 849.60 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-05-04 10:15:00 | 832.45 | 2026-05-07 09:15:00 | 845.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-05-05 09:15:00 | 836.30 | 2026-05-07 09:15:00 | 845.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-05-05 13:45:00 | 839.65 | 2026-05-07 09:15:00 | 845.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-05-05 15:15:00 | 838.00 | 2026-05-07 09:15:00 | 845.95 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-05-06 11:00:00 | 833.60 | 2026-05-07 09:15:00 | 845.95 | STOP_HIT | 1.00 | -1.48% |
