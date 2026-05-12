# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1261.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 193 |
| ALERT1 | 141 |
| ALERT2 | 139 |
| ALERT2_SKIP | 57 |
| ALERT3 | 342 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 170 |
| PARTIAL | 30 |
| TARGET_HIT | 16 |
| STOP_HIT | 165 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 211 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 89 / 122
- **Target hits / Stop hits / Partials:** 16 / 165 / 30
- **Avg / median % per leg:** 0.95% / -0.70%
- **Sum % (uncompounded):** 200.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 89 | 38 | 42.7% | 14 | 69 | 6 | 1.55% | 137.8% |
| BUY @ 2nd Alert (retest1) | 16 | 12 | 75.0% | 0 | 10 | 6 | 3.07% | 49.0% |
| BUY @ 3rd Alert (retest2) | 73 | 26 | 35.6% | 14 | 59 | 0 | 1.22% | 88.8% |
| SELL (all) | 122 | 51 | 41.8% | 2 | 96 | 24 | 0.51% | 62.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.52% | -0.5% |
| SELL @ 3rd Alert (retest2) | 121 | 51 | 42.1% | 2 | 95 | 24 | 0.52% | 63.1% |
| retest1 (combined) | 17 | 12 | 70.6% | 0 | 11 | 6 | 2.85% | 48.5% |
| retest2 (combined) | 194 | 77 | 39.7% | 16 | 154 | 24 | 0.78% | 151.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 14:15:00 | 495.83 | 496.64 | 496.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 12:15:00 | 495.20 | 496.25 | 496.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 10:15:00 | 495.13 | 491.78 | 493.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 495.13 | 491.78 | 493.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 495.13 | 491.78 | 493.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:45:00 | 495.13 | 491.78 | 493.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 495.00 | 492.42 | 493.23 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 14:15:00 | 495.18 | 493.76 | 493.72 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 489.95 | 493.26 | 493.51 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 495.30 | 493.72 | 493.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 15:15:00 | 493.25 | 494.17 | 494.21 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 495.85 | 494.50 | 494.36 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 493.45 | 494.45 | 494.51 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 10:15:00 | 494.93 | 494.08 | 494.07 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 10:15:00 | 493.28 | 494.17 | 494.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 13:15:00 | 492.25 | 493.43 | 493.82 | Break + close below crossover candle low |

### Cycle 10 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 515.28 | 497.32 | 495.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 521.05 | 502.07 | 497.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 14:15:00 | 534.15 | 534.98 | 527.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-02 15:15:00 | 535.00 | 534.98 | 527.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 531.00 | 534.19 | 528.20 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 517.50 | 525.66 | 526.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 513.60 | 523.25 | 525.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 519.28 | 518.03 | 521.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 523.83 | 518.70 | 519.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 523.83 | 518.70 | 519.79 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 12:15:00 | 522.98 | 520.90 | 520.66 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 09:15:00 | 516.75 | 520.09 | 520.40 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 522.00 | 519.15 | 519.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 524.23 | 522.10 | 521.13 | Break + close above crossover candle high |

### Cycle 15 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 504.95 | 519.10 | 519.96 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 14:15:00 | 521.10 | 516.70 | 516.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 521.90 | 518.45 | 517.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 10:15:00 | 518.00 | 518.36 | 517.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 10:15:00 | 518.00 | 518.36 | 517.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 518.00 | 518.36 | 517.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:00:00 | 518.00 | 518.36 | 517.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 518.95 | 518.48 | 517.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:45:00 | 519.05 | 518.48 | 517.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 517.25 | 518.23 | 517.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 13:00:00 | 517.25 | 518.23 | 517.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 13:15:00 | 517.73 | 518.13 | 517.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 13:30:00 | 515.85 | 518.13 | 517.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 517.38 | 517.98 | 517.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:45:00 | 516.58 | 517.98 | 517.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 517.78 | 517.94 | 517.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 520.98 | 517.94 | 517.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 516.50 | 518.72 | 518.46 | SL hit (close<static) qty=1.00 sl=516.65 alert=retest2 |

### Cycle 17 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 516.17 | 518.21 | 518.25 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 521.05 | 518.74 | 518.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 522.90 | 519.92 | 519.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 519.08 | 519.75 | 519.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 10:15:00 | 519.08 | 519.75 | 519.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 519.08 | 519.75 | 519.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 11:00:00 | 519.08 | 519.75 | 519.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 521.80 | 520.16 | 519.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 09:15:00 | 532.28 | 520.19 | 519.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 517.50 | 524.21 | 522.98 | SL hit (close<static) qty=1.00 sl=517.78 alert=retest2 |

### Cycle 19 — SELL (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 13:15:00 | 520.23 | 522.04 | 522.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 517.70 | 521.17 | 521.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 525.00 | 521.01 | 521.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 11:15:00 | 525.00 | 521.01 | 521.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 525.00 | 521.01 | 521.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:45:00 | 525.28 | 521.01 | 521.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 527.15 | 522.24 | 521.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 530.75 | 524.57 | 523.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 13:15:00 | 554.00 | 554.51 | 546.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 14:00:00 | 554.00 | 554.51 | 546.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 588.90 | 592.09 | 588.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:00:00 | 588.90 | 592.09 | 588.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 585.30 | 590.73 | 587.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 13:00:00 | 585.30 | 590.73 | 587.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 588.45 | 590.27 | 587.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:30:00 | 590.73 | 590.06 | 588.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:15:00 | 597.50 | 589.85 | 588.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 15:15:00 | 598.50 | 602.60 | 603.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 598.50 | 602.60 | 603.14 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 610.70 | 604.79 | 604.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 10:15:00 | 613.92 | 606.61 | 604.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 610.30 | 611.56 | 608.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 610.30 | 611.56 | 608.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 610.30 | 611.56 | 608.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 609.05 | 611.56 | 608.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 609.98 | 611.24 | 608.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 10:45:00 | 609.28 | 611.24 | 608.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 602.50 | 609.49 | 608.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 602.50 | 609.49 | 608.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 603.73 | 608.34 | 607.80 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 602.00 | 607.07 | 607.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 12:15:00 | 601.90 | 603.98 | 604.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 598.48 | 595.95 | 598.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 598.48 | 595.95 | 598.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 598.48 | 595.95 | 598.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 09:30:00 | 592.70 | 595.05 | 596.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 11:15:00 | 593.05 | 594.78 | 596.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 09:15:00 | 608.65 | 597.86 | 596.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 608.65 | 597.86 | 596.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 10:15:00 | 613.15 | 600.91 | 598.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 10:15:00 | 608.50 | 609.46 | 605.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 10:30:00 | 608.20 | 609.46 | 605.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 607.17 | 609.02 | 605.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:00:00 | 607.17 | 609.02 | 605.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 610.55 | 609.33 | 606.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:30:00 | 605.88 | 609.33 | 606.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 617.78 | 621.13 | 617.42 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 606.92 | 615.61 | 615.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 605.92 | 609.55 | 612.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 608.78 | 608.70 | 611.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 608.78 | 608.70 | 611.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 611.48 | 608.54 | 610.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:00:00 | 611.48 | 608.54 | 610.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 614.48 | 609.73 | 610.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 614.48 | 609.73 | 610.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 610.20 | 609.82 | 610.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:45:00 | 608.73 | 609.94 | 610.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 608.75 | 609.70 | 610.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:15:00 | 608.20 | 609.31 | 610.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 10:45:00 | 608.00 | 604.28 | 606.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 609.53 | 605.33 | 606.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 12:00:00 | 609.53 | 605.33 | 606.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 610.98 | 606.46 | 607.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:00:00 | 610.98 | 606.46 | 607.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-08 14:15:00 | 616.50 | 609.22 | 608.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 14:15:00 | 616.50 | 609.22 | 608.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 09:15:00 | 619.75 | 612.45 | 609.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 10:15:00 | 611.90 | 612.34 | 610.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-09 11:00:00 | 611.90 | 612.34 | 610.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 612.28 | 612.33 | 610.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 14:00:00 | 614.60 | 612.74 | 610.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 09:15:00 | 608.00 | 611.29 | 610.63 | SL hit (close<static) qty=1.00 sl=608.50 alert=retest2 |

### Cycle 27 — SELL (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 11:15:00 | 605.88 | 609.60 | 609.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 12:15:00 | 604.98 | 608.68 | 609.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 10:15:00 | 585.98 | 585.07 | 589.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 14:45:00 | 584.00 | 585.34 | 588.17 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 578.85 | 576.04 | 578.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:45:00 | 580.90 | 576.04 | 578.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 577.13 | 576.25 | 578.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 15:15:00 | 575.00 | 576.25 | 578.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 587.03 | 578.21 | 579.16 | SL hit (close>ema400) qty=1.00 sl=579.16 alert=retest1 |

### Cycle 28 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 584.28 | 580.39 | 580.04 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 578.83 | 581.83 | 581.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 577.03 | 580.37 | 581.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 11:15:00 | 567.45 | 566.47 | 569.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 12:00:00 | 567.45 | 566.47 | 569.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 573.03 | 567.27 | 568.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:00:00 | 573.03 | 567.27 | 568.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 576.63 | 569.14 | 569.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:30:00 | 580.85 | 569.14 | 569.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 575.10 | 570.34 | 570.10 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 565.48 | 571.14 | 571.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 10:15:00 | 564.20 | 569.75 | 570.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 13:15:00 | 571.00 | 568.40 | 569.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 13:15:00 | 571.00 | 568.40 | 569.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 571.00 | 568.40 | 569.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:00:00 | 571.00 | 568.40 | 569.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 571.40 | 569.00 | 569.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 15:15:00 | 569.50 | 569.00 | 569.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 577.83 | 570.85 | 570.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 577.83 | 570.85 | 570.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 606.40 | 582.23 | 576.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 12:15:00 | 608.50 | 608.67 | 602.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 12:30:00 | 609.88 | 608.67 | 602.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 601.55 | 607.06 | 602.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 15:00:00 | 601.55 | 607.06 | 602.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 601.98 | 606.04 | 602.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:15:00 | 606.53 | 606.04 | 602.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-11 09:15:00 | 667.18 | 636.73 | 621.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 656.85 | 665.65 | 666.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 646.98 | 655.86 | 660.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 15:15:00 | 649.50 | 647.17 | 653.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 09:15:00 | 651.00 | 647.17 | 653.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 645.67 | 646.87 | 652.56 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 10:15:00 | 657.70 | 654.93 | 654.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 671.00 | 659.09 | 656.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 662.03 | 662.84 | 659.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 662.03 | 662.84 | 659.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 667.13 | 673.28 | 670.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 667.13 | 673.28 | 670.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 669.00 | 672.43 | 670.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 665.65 | 671.07 | 670.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 666.18 | 670.09 | 669.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:30:00 | 664.80 | 670.09 | 669.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 671.23 | 674.55 | 672.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:30:00 | 673.23 | 674.55 | 672.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 670.13 | 673.67 | 672.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:45:00 | 669.50 | 673.67 | 672.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 671.03 | 673.14 | 672.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:15:00 | 669.43 | 673.14 | 672.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 668.35 | 671.18 | 671.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 659.00 | 668.55 | 670.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 661.80 | 661.32 | 664.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 661.80 | 661.32 | 664.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 661.80 | 661.32 | 664.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 658.15 | 661.32 | 664.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 13:00:00 | 658.55 | 660.37 | 663.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 09:15:00 | 681.48 | 664.44 | 664.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 681.48 | 664.44 | 664.34 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 660.03 | 666.35 | 667.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 13:15:00 | 653.90 | 662.14 | 664.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 666.58 | 660.46 | 663.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 666.58 | 660.46 | 663.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 666.58 | 660.46 | 663.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 665.28 | 660.46 | 663.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 662.13 | 660.79 | 663.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 11:30:00 | 660.55 | 660.77 | 662.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:30:00 | 659.65 | 660.61 | 662.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 14:30:00 | 660.58 | 660.44 | 662.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 15:15:00 | 659.50 | 660.44 | 662.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 673.50 | 662.90 | 663.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 673.50 | 662.90 | 663.00 | SL hit (close>static) qty=1.00 sl=667.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 669.50 | 664.22 | 663.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 697.15 | 675.43 | 669.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 694.68 | 694.80 | 686.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 15:00:00 | 694.68 | 694.80 | 686.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 695.55 | 694.66 | 687.49 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 677.30 | 686.60 | 686.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 674.48 | 678.98 | 682.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 681.00 | 672.57 | 675.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 681.00 | 672.57 | 675.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 681.00 | 672.57 | 675.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:30:00 | 678.68 | 672.57 | 675.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 676.75 | 673.41 | 675.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:15:00 | 674.68 | 673.41 | 675.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 681.48 | 675.27 | 675.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-23 10:15:00 | 681.48 | 675.27 | 675.22 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 674.83 | 675.18 | 675.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 667.88 | 673.72 | 674.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 670.65 | 666.79 | 670.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 670.65 | 666.79 | 670.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 670.65 | 666.79 | 670.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:15:00 | 671.00 | 666.79 | 670.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 671.58 | 667.75 | 670.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:45:00 | 680.00 | 667.75 | 670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 665.03 | 667.20 | 670.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:30:00 | 661.88 | 664.75 | 668.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 649.80 | 663.22 | 666.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 13:15:00 | 664.28 | 659.64 | 659.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 664.28 | 659.64 | 659.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 696.48 | 668.41 | 663.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 12:15:00 | 757.50 | 758.48 | 738.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-02 12:45:00 | 759.83 | 758.48 | 738.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 874.75 | 904.73 | 897.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 904.88 | 895.28 | 894.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 14:15:00 | 888.40 | 894.11 | 894.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 14:15:00 | 888.40 | 894.11 | 894.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 15:15:00 | 887.35 | 892.76 | 893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 14:15:00 | 876.25 | 875.84 | 883.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-20 15:00:00 | 876.25 | 875.84 | 883.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 864.00 | 872.54 | 880.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 14:00:00 | 856.95 | 865.90 | 871.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 904.00 | 871.59 | 872.29 | SL hit (close>static) qty=1.00 sl=882.70 alert=retest2 |

### Cycle 44 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 892.40 | 875.75 | 874.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 911.98 | 900.40 | 892.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 15:15:00 | 945.00 | 946.20 | 933.38 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 09:15:00 | 958.50 | 946.20 | 933.38 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 15:00:00 | 950.45 | 951.98 | 942.72 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 949.95 | 960.40 | 955.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 949.95 | 960.40 | 955.15 | SL hit (close<ema400) qty=1.00 sl=955.15 alert=retest1 |

### Cycle 45 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 948.00 | 954.25 | 954.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 943.65 | 950.54 | 952.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 15:15:00 | 934.88 | 934.11 | 940.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 09:15:00 | 946.85 | 934.11 | 940.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 947.10 | 936.71 | 940.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 953.30 | 936.71 | 940.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 945.50 | 938.47 | 941.15 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 946.00 | 942.78 | 942.63 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 937.80 | 941.75 | 942.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 931.48 | 939.16 | 940.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 14:15:00 | 938.68 | 938.12 | 940.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-12 15:00:00 | 938.68 | 938.12 | 940.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 950.50 | 940.34 | 940.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 947.30 | 940.34 | 940.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 10:15:00 | 967.28 | 945.72 | 943.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 11:15:00 | 977.78 | 952.14 | 946.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 13:15:00 | 970.25 | 971.77 | 963.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 15:00:00 | 975.00 | 972.41 | 964.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 960.25 | 969.92 | 964.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-15 09:15:00 | 960.25 | 969.92 | 964.44 | SL hit (close<ema400) qty=1.00 sl=964.44 alert=retest1 |

### Cycle 49 — SELL (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 12:15:00 | 948.00 | 960.27 | 960.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 14:15:00 | 945.15 | 955.67 | 958.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 10:15:00 | 953.90 | 953.05 | 956.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-18 11:00:00 | 953.90 | 953.05 | 956.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 959.03 | 954.24 | 956.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:00:00 | 959.03 | 954.24 | 956.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 955.98 | 954.59 | 956.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 13:45:00 | 952.78 | 954.02 | 956.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 14:45:00 | 952.93 | 954.14 | 954.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 09:15:00 | 958.70 | 955.35 | 954.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 958.70 | 955.35 | 954.91 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 917.13 | 948.74 | 952.17 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 927.93 | 926.50 | 926.34 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 12:15:00 | 921.53 | 925.51 | 925.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 10:15:00 | 918.53 | 923.47 | 924.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 910.98 | 910.82 | 913.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 920.38 | 912.40 | 913.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 920.38 | 912.40 | 913.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 917.48 | 912.40 | 913.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 10:15:00 | 919.45 | 913.81 | 913.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 10:15:00 | 925.53 | 920.26 | 917.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 936.00 | 942.33 | 938.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 14:15:00 | 936.00 | 942.33 | 938.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 936.00 | 942.33 | 938.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:30:00 | 934.48 | 942.33 | 938.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 941.00 | 942.06 | 938.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 943.35 | 942.06 | 938.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-09 13:15:00 | 931.63 | 937.92 | 937.81 | SL hit (close<static) qty=1.00 sl=933.50 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 15:15:00 | 936.00 | 937.55 | 937.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 932.35 | 936.51 | 937.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 928.10 | 925.09 | 928.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 928.10 | 925.09 | 928.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 928.10 | 925.09 | 928.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:45:00 | 927.50 | 925.09 | 928.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 923.25 | 924.72 | 927.87 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 941.00 | 928.78 | 927.87 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 924.83 | 930.14 | 930.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 899.48 | 921.72 | 926.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 915.75 | 913.99 | 919.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 915.75 | 913.99 | 919.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 915.75 | 913.99 | 919.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:00:00 | 911.55 | 913.50 | 918.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 09:30:00 | 911.33 | 910.77 | 914.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 14:15:00 | 865.97 | 887.08 | 898.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 14:15:00 | 865.76 | 887.08 | 898.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-24 14:15:00 | 881.95 | 877.59 | 886.83 | SL hit (close>ema200) qty=0.50 sl=877.59 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 887.95 | 884.82 | 884.63 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 15:15:00 | 881.98 | 884.11 | 884.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 11:15:00 | 876.83 | 882.02 | 883.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 12:15:00 | 878.23 | 874.85 | 877.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 12:15:00 | 878.23 | 874.85 | 877.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 878.23 | 874.85 | 877.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:45:00 | 876.88 | 874.85 | 877.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 882.65 | 876.41 | 878.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:45:00 | 883.65 | 876.41 | 878.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 885.55 | 878.24 | 878.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 14:45:00 | 890.68 | 878.24 | 878.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 15:15:00 | 885.48 | 879.69 | 879.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 10:15:00 | 917.18 | 888.51 | 883.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 13:15:00 | 928.65 | 931.25 | 920.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 14:00:00 | 928.65 | 931.25 | 920.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 913.00 | 927.60 | 919.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 907.38 | 927.60 | 919.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 916.00 | 925.28 | 919.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 912.45 | 925.28 | 919.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 927.50 | 923.56 | 919.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 939.63 | 926.08 | 922.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 10:15:00 | 948.60 | 966.75 | 968.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 948.60 | 966.75 | 968.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 930.00 | 949.67 | 959.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 12:15:00 | 940.80 | 938.00 | 948.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 12:45:00 | 943.73 | 938.00 | 948.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 942.63 | 938.92 | 948.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:45:00 | 945.50 | 938.92 | 948.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 945.28 | 940.20 | 947.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 945.28 | 940.20 | 947.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 939.50 | 940.06 | 947.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:30:00 | 932.43 | 940.60 | 946.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 13:15:00 | 947.65 | 944.15 | 946.60 | SL hit (close>static) qty=1.00 sl=947.50 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 13:15:00 | 931.50 | 918.39 | 916.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 949.50 | 927.79 | 921.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 955.48 | 957.50 | 946.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 14:00:00 | 955.48 | 957.50 | 946.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 952.53 | 963.36 | 958.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 968.50 | 960.12 | 958.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 10:00:00 | 967.80 | 961.66 | 959.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 10:15:00 | 953.50 | 958.53 | 959.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 10:15:00 | 953.50 | 958.53 | 959.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 11:15:00 | 948.88 | 956.60 | 958.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 956.00 | 955.74 | 957.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-04 14:00:00 | 956.00 | 955.74 | 957.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 919.40 | 919.27 | 925.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:30:00 | 920.00 | 919.27 | 925.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 881.98 | 900.86 | 909.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 866.18 | 895.08 | 905.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:30:00 | 866.03 | 858.01 | 869.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 822.87 | 838.57 | 846.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 822.73 | 838.57 | 846.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 865.63 | 837.55 | 840.55 | SL hit (close>ema200) qty=0.50 sl=837.55 alert=retest2 |

### Cycle 64 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 871.00 | 844.24 | 843.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 875.88 | 854.35 | 848.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 894.33 | 895.29 | 884.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 14:30:00 | 894.93 | 895.29 | 884.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 851.00 | 885.98 | 882.08 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 854.45 | 874.48 | 877.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 13:15:00 | 852.28 | 866.80 | 873.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 862.45 | 858.69 | 866.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-28 10:30:00 | 862.93 | 858.69 | 866.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 886.88 | 864.16 | 865.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 886.88 | 864.16 | 865.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 894.90 | 870.31 | 868.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 897.73 | 875.79 | 870.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 11:15:00 | 933.70 | 934.75 | 924.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 12:00:00 | 933.70 | 934.75 | 924.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 935.93 | 936.53 | 929.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:45:00 | 927.03 | 936.53 | 929.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 935.50 | 937.10 | 934.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:00:00 | 935.50 | 937.10 | 934.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 939.68 | 937.62 | 934.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:30:00 | 933.88 | 937.62 | 934.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 931.60 | 936.75 | 935.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 09:45:00 | 931.40 | 936.75 | 935.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 930.65 | 935.53 | 934.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 11:15:00 | 935.73 | 935.53 | 934.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 14:45:00 | 936.83 | 934.92 | 934.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 15:15:00 | 932.00 | 934.34 | 934.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 15:15:00 | 932.00 | 934.34 | 934.49 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 939.80 | 935.43 | 934.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 10:15:00 | 951.45 | 938.63 | 936.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 14:15:00 | 940.90 | 941.92 | 939.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-10 15:00:00 | 940.90 | 941.92 | 939.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 940.50 | 941.63 | 939.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 963.70 | 941.63 | 939.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 10:15:00 | 968.83 | 984.11 | 985.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 968.83 | 984.11 | 985.91 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 996.98 | 983.67 | 983.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 1002.83 | 987.50 | 984.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 1040.00 | 1042.87 | 1028.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 15:00:00 | 1040.00 | 1042.87 | 1028.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1044.93 | 1043.14 | 1031.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 1065.00 | 1047.94 | 1035.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 12:00:00 | 1060.58 | 1062.03 | 1050.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:30:00 | 1060.50 | 1054.84 | 1052.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 1061.00 | 1054.89 | 1053.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1054.50 | 1054.82 | 1053.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 1054.50 | 1054.82 | 1053.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 1049.18 | 1053.69 | 1052.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:00:00 | 1049.18 | 1053.69 | 1052.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 1051.78 | 1053.31 | 1052.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 1040.68 | 1051.50 | 1052.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 1040.68 | 1051.50 | 1052.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 1033.88 | 1041.16 | 1045.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 1040.40 | 1034.81 | 1040.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 14:15:00 | 1040.40 | 1034.81 | 1040.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 1040.40 | 1034.81 | 1040.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 1040.40 | 1034.81 | 1040.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 1040.95 | 1036.03 | 1040.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 1076.75 | 1036.03 | 1040.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 1065.47 | 1041.92 | 1042.52 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 10:15:00 | 1083.15 | 1050.17 | 1046.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 1106.50 | 1074.67 | 1061.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1069.50 | 1075.00 | 1064.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 11:15:00 | 1069.50 | 1075.00 | 1064.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 1069.50 | 1075.00 | 1064.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:45:00 | 1067.38 | 1075.00 | 1064.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1059.13 | 1071.83 | 1063.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 1059.13 | 1071.83 | 1063.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 1067.53 | 1070.97 | 1064.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 10:00:00 | 1078.72 | 1068.51 | 1064.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 11:30:00 | 1074.10 | 1069.94 | 1065.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:00:00 | 1073.88 | 1069.94 | 1065.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:30:00 | 1074.22 | 1071.20 | 1066.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1060.00 | 1072.27 | 1069.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 1060.00 | 1072.27 | 1069.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 1061.47 | 1070.11 | 1068.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:30:00 | 1062.63 | 1070.11 | 1068.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 1052.50 | 1066.59 | 1067.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 1052.50 | 1066.59 | 1067.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 1036.00 | 1057.98 | 1063.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 15:15:00 | 997.50 | 997.43 | 1013.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 09:15:00 | 1014.95 | 997.43 | 1013.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1004.38 | 998.82 | 1012.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 1018.35 | 998.82 | 1012.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 1011.40 | 1003.16 | 1011.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:00:00 | 1011.40 | 1003.16 | 1011.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1017.48 | 1006.03 | 1011.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:45:00 | 1019.00 | 1006.03 | 1011.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1026.95 | 1010.21 | 1013.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 1026.95 | 1010.21 | 1013.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1058.50 | 1022.72 | 1018.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 14:15:00 | 1075.00 | 1065.85 | 1058.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 13:15:00 | 1077.55 | 1081.39 | 1075.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 13:45:00 | 1079.47 | 1081.39 | 1075.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1085.25 | 1082.17 | 1076.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 1076.83 | 1082.17 | 1076.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1068.53 | 1079.97 | 1076.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1068.53 | 1079.97 | 1076.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1066.53 | 1077.28 | 1075.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 1066.00 | 1077.28 | 1075.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1078.00 | 1075.77 | 1075.30 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 1070.00 | 1074.31 | 1074.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 1058.63 | 1071.17 | 1073.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 1067.78 | 1066.49 | 1069.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:00:00 | 1067.78 | 1066.49 | 1069.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1029.88 | 1034.30 | 1040.37 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1061.50 | 1041.21 | 1040.58 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 992.03 | 1051.11 | 1051.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 952.30 | 1031.35 | 1042.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1019.00 | 994.93 | 1006.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1019.00 | 994.93 | 1006.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1019.00 | 994.93 | 1006.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1017.23 | 994.93 | 1006.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1022.15 | 1000.38 | 1008.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:15:00 | 1022.58 | 1000.38 | 1008.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 1019.33 | 1011.75 | 1011.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1043.58 | 1018.12 | 1014.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1032.70 | 1036.54 | 1031.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 1032.70 | 1036.54 | 1031.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1032.70 | 1036.54 | 1031.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 1033.00 | 1036.54 | 1031.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1035.00 | 1036.23 | 1031.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1037.85 | 1036.23 | 1031.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1046.25 | 1038.23 | 1032.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 1053.50 | 1040.44 | 1034.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1051.30 | 1044.66 | 1039.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:15:00 | 1049.20 | 1049.70 | 1043.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:45:00 | 1049.47 | 1049.26 | 1044.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1048.03 | 1049.01 | 1044.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:15:00 | 1048.97 | 1049.01 | 1044.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1048.97 | 1049.00 | 1044.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1050.83 | 1049.00 | 1044.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 1043.08 | 1051.26 | 1051.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 1043.08 | 1051.26 | 1051.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 12:15:00 | 1039.95 | 1049.00 | 1050.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 11:15:00 | 1024.53 | 1023.28 | 1028.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 11:30:00 | 1025.55 | 1023.28 | 1028.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1016.50 | 1016.69 | 1020.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 1013.18 | 1015.98 | 1019.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:00:00 | 1015.23 | 1015.82 | 1018.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:45:00 | 1015.00 | 1015.53 | 1018.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 1015.10 | 1015.88 | 1018.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1016.03 | 1015.91 | 1017.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:30:00 | 1016.00 | 1015.91 | 1017.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1014.50 | 1015.63 | 1017.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:30:00 | 1013.50 | 1015.23 | 1017.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 1013.65 | 1015.23 | 1017.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:45:00 | 1012.75 | 1014.48 | 1016.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1089.75 | 1023.03 | 1017.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 1089.75 | 1023.03 | 1017.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 1122.30 | 1042.89 | 1026.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 1197.55 | 1197.99 | 1155.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 14:00:00 | 1209.97 | 1199.35 | 1166.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1162.03 | 1190.55 | 1170.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 1162.03 | 1190.55 | 1170.56 | SL hit (close<ema400) qty=1.00 sl=1170.56 alert=retest1 |

### Cycle 81 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 1161.00 | 1165.21 | 1165.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 1156.13 | 1162.94 | 1164.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 11:15:00 | 1163.50 | 1163.00 | 1164.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 11:15:00 | 1163.50 | 1163.00 | 1164.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1163.50 | 1163.00 | 1164.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:30:00 | 1167.38 | 1163.00 | 1164.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1158.50 | 1162.10 | 1163.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:30:00 | 1154.60 | 1160.42 | 1162.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:00:00 | 1153.55 | 1158.34 | 1160.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 1155.90 | 1158.00 | 1160.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 13:15:00 | 1155.88 | 1158.00 | 1160.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1167.05 | 1158.68 | 1159.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 1167.05 | 1158.68 | 1159.96 | SL hit (close>static) qty=1.00 sl=1164.47 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 1179.97 | 1155.84 | 1155.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 11:15:00 | 1211.78 | 1172.24 | 1163.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 1204.50 | 1208.91 | 1192.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 13:00:00 | 1204.50 | 1208.91 | 1192.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1234.78 | 1214.08 | 1200.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 1200.00 | 1214.08 | 1200.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1199.58 | 1209.72 | 1201.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:30:00 | 1199.50 | 1209.72 | 1201.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1200.20 | 1207.81 | 1201.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 1200.20 | 1207.81 | 1201.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1203.13 | 1206.88 | 1201.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 1206.50 | 1206.88 | 1201.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1198.50 | 1205.14 | 1201.71 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1191.93 | 1198.29 | 1199.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 1181.28 | 1193.12 | 1196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 1150.00 | 1145.82 | 1152.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 12:00:00 | 1150.00 | 1145.82 | 1152.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1142.43 | 1145.15 | 1151.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 1148.50 | 1145.15 | 1151.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1143.80 | 1144.88 | 1150.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:15:00 | 1155.83 | 1144.88 | 1150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1159.53 | 1147.81 | 1151.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 1167.35 | 1147.81 | 1151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1161.72 | 1150.59 | 1152.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1172.80 | 1150.59 | 1152.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1191.00 | 1158.67 | 1156.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 1198.58 | 1166.65 | 1159.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1180.60 | 1183.57 | 1173.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1180.60 | 1183.57 | 1173.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1180.60 | 1183.57 | 1173.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 1203.43 | 1187.91 | 1177.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 1221.45 | 1235.03 | 1235.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 1221.45 | 1235.03 | 1235.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1214.00 | 1225.41 | 1230.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 12:15:00 | 1228.53 | 1225.53 | 1229.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 12:15:00 | 1228.53 | 1225.53 | 1229.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 1228.53 | 1225.53 | 1229.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:30:00 | 1229.95 | 1225.53 | 1229.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 1226.60 | 1225.75 | 1228.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1186.05 | 1226.18 | 1228.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1235.05 | 1186.98 | 1181.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 1235.05 | 1186.98 | 1181.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1281.20 | 1228.32 | 1207.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 1286.15 | 1287.35 | 1263.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 14:45:00 | 1289.78 | 1287.35 | 1263.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1262.80 | 1276.28 | 1268.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1262.80 | 1276.28 | 1268.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1263.95 | 1273.81 | 1268.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1275.00 | 1273.81 | 1268.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:45:00 | 1273.68 | 1273.85 | 1268.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:45:00 | 1264.68 | 1270.51 | 1268.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 1264.72 | 1268.15 | 1267.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-16 13:15:00 | 1391.15 | 1322.13 | 1297.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 1502.00 | 1519.93 | 1520.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 1497.70 | 1512.23 | 1516.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1465.15 | 1450.95 | 1472.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 10:00:00 | 1465.15 | 1450.95 | 1472.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1431.60 | 1425.30 | 1435.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:30:00 | 1431.00 | 1425.30 | 1435.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1441.00 | 1427.19 | 1433.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 1441.00 | 1427.19 | 1433.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 1427.20 | 1427.19 | 1432.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:15:00 | 1424.65 | 1427.19 | 1432.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 1425.25 | 1427.44 | 1431.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1424.00 | 1425.94 | 1429.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:15:00 | 1424.50 | 1425.94 | 1429.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1425.40 | 1425.83 | 1429.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:30:00 | 1419.85 | 1422.99 | 1426.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:00:00 | 1413.00 | 1419.87 | 1424.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 1353.42 | 1369.93 | 1386.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 1353.99 | 1369.93 | 1386.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 1352.80 | 1369.93 | 1386.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 1353.27 | 1369.93 | 1386.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 12:15:00 | 1348.86 | 1366.64 | 1383.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 1363.60 | 1361.56 | 1375.26 | SL hit (close>ema200) qty=0.50 sl=1361.56 alert=retest2 |

### Cycle 88 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 1382.00 | 1371.82 | 1371.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1395.30 | 1376.52 | 1373.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 1492.30 | 1495.31 | 1472.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 11:00:00 | 1492.30 | 1495.31 | 1472.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1503.00 | 1496.85 | 1475.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 1480.45 | 1496.85 | 1475.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1523.00 | 1525.01 | 1517.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1513.05 | 1525.01 | 1517.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1515.00 | 1523.01 | 1517.26 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 1501.00 | 1512.15 | 1513.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 1498.45 | 1509.41 | 1511.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 09:15:00 | 1517.70 | 1507.69 | 1510.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 1517.70 | 1507.69 | 1510.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1517.70 | 1507.69 | 1510.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 1520.70 | 1507.69 | 1510.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1506.15 | 1507.38 | 1509.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 11:30:00 | 1502.05 | 1505.67 | 1508.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:15:00 | 1426.95 | 1441.61 | 1455.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 15:15:00 | 1441.10 | 1438.28 | 1448.55 | SL hit (close>ema200) qty=0.50 sl=1438.28 alert=retest2 |

### Cycle 90 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1437.80 | 1387.28 | 1386.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 1463.05 | 1402.43 | 1393.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 14:15:00 | 1477.75 | 1477.93 | 1461.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1507.10 | 1477.95 | 1463.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 12:15:00 | 1582.45 | 1520.67 | 1489.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 1576.30 | 1582.06 | 1557.33 | SL hit (close<ema200) qty=0.50 sl=1582.06 alert=retest1 |

### Cycle 91 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1545.50 | 1561.30 | 1561.77 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1583.00 | 1564.53 | 1562.49 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 1554.40 | 1563.44 | 1564.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1547.00 | 1560.15 | 1562.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 1497.75 | 1489.44 | 1510.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 1497.75 | 1489.44 | 1510.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1475.30 | 1437.81 | 1454.35 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 1488.20 | 1465.56 | 1464.49 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 1455.00 | 1465.31 | 1466.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 13:15:00 | 1450.00 | 1462.25 | 1464.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 1463.00 | 1462.03 | 1464.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 15:15:00 | 1463.00 | 1462.03 | 1464.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 1463.00 | 1462.03 | 1464.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 1488.60 | 1462.03 | 1464.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1493.70 | 1468.36 | 1466.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 1519.10 | 1488.13 | 1477.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1515.90 | 1533.60 | 1515.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1515.90 | 1533.60 | 1515.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1515.90 | 1533.60 | 1515.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1515.90 | 1533.60 | 1515.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1507.85 | 1528.45 | 1514.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1507.85 | 1528.45 | 1514.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1517.20 | 1526.20 | 1514.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 1526.80 | 1526.32 | 1515.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 1524.00 | 1524.37 | 1517.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:00:00 | 1524.45 | 1524.38 | 1518.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 1521.30 | 1522.64 | 1518.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1544.55 | 1527.02 | 1521.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1587.25 | 1532.85 | 1525.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-08 10:15:00 | 1676.40 | 1614.30 | 1588.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 1563.00 | 1584.79 | 1587.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1553.00 | 1578.43 | 1584.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1570.20 | 1569.51 | 1578.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1570.20 | 1569.51 | 1578.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1570.20 | 1569.51 | 1578.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:15:00 | 1558.35 | 1569.51 | 1578.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1480.43 | 1528.12 | 1551.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 1492.00 | 1475.72 | 1507.14 | SL hit (close>ema200) qty=0.50 sl=1475.72 alert=retest2 |

### Cycle 98 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1533.85 | 1502.38 | 1498.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 14:15:00 | 1546.85 | 1525.54 | 1515.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 15:15:00 | 1535.00 | 1536.85 | 1528.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1563.40 | 1536.85 | 1528.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:15:00 | 1641.57 | 1613.58 | 1591.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 1599.15 | 1612.65 | 1597.08 | SL hit (close<ema200) qty=0.50 sl=1612.65 alert=retest1 |

### Cycle 99 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 1939.40 | 1949.91 | 1950.61 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 1960.80 | 1951.66 | 1950.63 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 1930.00 | 1948.92 | 1949.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 11:15:00 | 1913.90 | 1939.94 | 1945.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 1814.35 | 1804.10 | 1824.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 1814.35 | 1804.10 | 1824.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1781.75 | 1799.78 | 1819.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:30:00 | 1777.20 | 1796.48 | 1816.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 1774.00 | 1791.76 | 1812.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1776.70 | 1784.95 | 1803.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 1778.65 | 1782.74 | 1797.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1757.90 | 1752.38 | 1765.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 1764.15 | 1752.38 | 1765.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1791.60 | 1761.51 | 1767.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 1791.60 | 1761.51 | 1767.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1813.90 | 1771.99 | 1772.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 1808.55 | 1771.99 | 1772.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1804.40 | 1778.47 | 1774.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1804.40 | 1778.47 | 1774.95 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1753.00 | 1790.58 | 1792.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1735.35 | 1769.42 | 1781.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1763.00 | 1750.71 | 1764.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 12:15:00 | 1763.00 | 1750.71 | 1764.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1763.00 | 1750.71 | 1764.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 1763.00 | 1750.71 | 1764.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1759.30 | 1752.43 | 1764.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 1764.00 | 1752.43 | 1764.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1760.35 | 1754.02 | 1763.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:30:00 | 1757.50 | 1754.02 | 1763.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1710.15 | 1745.40 | 1758.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 1696.00 | 1716.29 | 1732.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 1611.20 | 1649.19 | 1683.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 1526.40 | 1569.25 | 1617.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 104 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 1603.00 | 1599.73 | 1599.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1608.85 | 1601.55 | 1600.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 1591.30 | 1601.59 | 1600.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 11:15:00 | 1591.30 | 1601.59 | 1600.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 1591.30 | 1601.59 | 1600.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 1591.30 | 1601.59 | 1600.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 1608.05 | 1602.88 | 1601.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 1610.90 | 1605.05 | 1602.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 12:15:00 | 1601.00 | 1601.64 | 1601.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1601.00 | 1601.64 | 1601.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 1592.55 | 1599.82 | 1600.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 1599.05 | 1598.72 | 1600.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 1599.05 | 1598.72 | 1600.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1599.05 | 1598.72 | 1600.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1583.00 | 1598.77 | 1599.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1503.85 | 1536.04 | 1561.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 1515.30 | 1508.74 | 1530.96 | SL hit (close>ema200) qty=0.50 sl=1508.74 alert=retest2 |

### Cycle 106 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1336.95 | 1308.76 | 1307.19 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1258.85 | 1298.88 | 1303.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 1252.65 | 1273.21 | 1288.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1266.15 | 1264.83 | 1280.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 1275.90 | 1264.83 | 1280.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 1281.45 | 1269.69 | 1277.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 1286.35 | 1269.69 | 1277.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 1291.40 | 1274.03 | 1279.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 1291.40 | 1274.03 | 1279.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1293.00 | 1277.83 | 1280.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 1342.00 | 1277.83 | 1280.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 1341.00 | 1290.46 | 1285.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 1355.35 | 1312.39 | 1297.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 1337.15 | 1338.66 | 1321.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 1337.15 | 1338.66 | 1321.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1328.85 | 1334.44 | 1323.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 1327.25 | 1334.44 | 1323.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1345.75 | 1335.57 | 1325.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:00:00 | 1354.00 | 1339.26 | 1328.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1303.00 | 1329.77 | 1328.47 | SL hit (close<static) qty=1.00 sl=1322.15 alert=retest2 |

### Cycle 109 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1305.40 | 1324.90 | 1326.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1295.00 | 1318.92 | 1323.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1256.20 | 1239.33 | 1262.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1256.20 | 1239.33 | 1262.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1257.00 | 1241.06 | 1255.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1257.00 | 1241.06 | 1255.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1259.90 | 1244.83 | 1255.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1259.90 | 1244.83 | 1255.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1255.00 | 1246.86 | 1255.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 1244.15 | 1246.32 | 1254.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1181.94 | 1207.97 | 1224.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 1203.80 | 1202.59 | 1216.08 | SL hit (close>ema200) qty=0.50 sl=1202.59 alert=retest2 |

### Cycle 110 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1215.00 | 1205.08 | 1204.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 1235.50 | 1212.10 | 1207.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1250.10 | 1252.41 | 1239.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 1239.60 | 1252.41 | 1239.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1227.80 | 1247.49 | 1238.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 1227.80 | 1247.49 | 1238.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1224.40 | 1242.87 | 1237.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 1220.95 | 1242.87 | 1237.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 1217.40 | 1232.73 | 1233.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 1211.75 | 1225.62 | 1229.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1114.30 | 1110.01 | 1136.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1114.30 | 1110.01 | 1136.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1120.95 | 1114.69 | 1121.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 1126.00 | 1114.69 | 1121.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1125.75 | 1116.91 | 1121.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 1125.75 | 1116.91 | 1121.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1134.55 | 1120.43 | 1123.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 1134.55 | 1120.43 | 1123.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1132.70 | 1122.89 | 1123.89 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1131.70 | 1124.65 | 1124.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 1137.40 | 1127.20 | 1125.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1161.95 | 1168.73 | 1155.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1161.95 | 1168.73 | 1155.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1151.85 | 1165.35 | 1155.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 1151.85 | 1165.35 | 1155.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1155.75 | 1163.43 | 1155.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 1148.70 | 1163.43 | 1155.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 1155.85 | 1161.92 | 1155.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 1155.85 | 1161.92 | 1155.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1152.00 | 1159.93 | 1155.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1135.95 | 1159.93 | 1155.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1134.90 | 1154.93 | 1153.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1134.90 | 1154.93 | 1153.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 1132.45 | 1150.43 | 1151.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 1125.25 | 1145.39 | 1149.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 1112.55 | 1106.41 | 1119.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 1102.05 | 1106.41 | 1119.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1089.70 | 1103.07 | 1116.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 1086.00 | 1100.86 | 1114.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 1084.80 | 1093.47 | 1107.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 14:30:00 | 1087.40 | 1092.98 | 1105.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1073.00 | 1092.55 | 1104.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1094.40 | 1069.85 | 1077.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1094.40 | 1069.85 | 1077.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1108.85 | 1077.65 | 1080.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 1108.85 | 1077.65 | 1080.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 1112.00 | 1084.52 | 1083.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 1112.00 | 1084.52 | 1083.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1121.30 | 1099.43 | 1090.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1218.40 | 1228.53 | 1208.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1218.40 | 1228.53 | 1208.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1210.50 | 1223.51 | 1209.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1213.95 | 1223.51 | 1209.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1210.10 | 1220.83 | 1209.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1210.00 | 1220.83 | 1209.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1220.80 | 1220.82 | 1210.67 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1198.00 | 1207.62 | 1207.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1181.60 | 1201.01 | 1204.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1190.95 | 1188.67 | 1195.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:45:00 | 1193.95 | 1188.67 | 1195.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 116 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1245.05 | 1199.73 | 1198.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 1248.25 | 1216.84 | 1207.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 15:15:00 | 1220.00 | 1220.50 | 1212.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:15:00 | 1212.00 | 1220.50 | 1212.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1209.45 | 1218.29 | 1212.08 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1201.35 | 1208.85 | 1209.02 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 1220.80 | 1210.05 | 1209.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1230.60 | 1219.54 | 1214.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1198.00 | 1223.67 | 1220.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1198.00 | 1223.67 | 1220.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1198.00 | 1223.67 | 1220.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1198.00 | 1223.67 | 1220.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1211.85 | 1221.31 | 1219.45 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1203.75 | 1217.80 | 1218.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1197.45 | 1211.26 | 1214.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1152.35 | 1150.76 | 1174.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1198.30 | 1150.76 | 1174.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1166.45 | 1153.90 | 1173.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1161.55 | 1168.61 | 1173.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 14:15:00 | 1168.35 | 1166.81 | 1166.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 1168.35 | 1166.81 | 1166.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 1171.45 | 1167.74 | 1167.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1225.40 | 1227.06 | 1213.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 12:15:00 | 1247.50 | 1230.01 | 1216.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 13:30:00 | 1244.90 | 1234.51 | 1221.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 14:15:00 | 1243.90 | 1234.51 | 1221.43 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 1276.60 | 1236.79 | 1224.81 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 15:15:00 | 1309.88 | 1285.65 | 1259.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 15:15:00 | 1307.15 | 1285.65 | 1259.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 15:15:00 | 1306.10 | 1285.65 | 1259.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1308.10 | 1309.72 | 1289.82 | SL hit (close<ema200) qty=0.50 sl=1309.72 alert=retest1 |

### Cycle 121 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1322.00 | 1339.18 | 1339.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1315.10 | 1326.46 | 1332.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1332.00 | 1327.57 | 1332.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1332.00 | 1327.57 | 1332.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1332.00 | 1327.57 | 1332.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1332.00 | 1327.57 | 1332.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1328.00 | 1327.66 | 1332.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 1320.00 | 1327.66 | 1332.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 1320.90 | 1326.30 | 1331.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:30:00 | 1323.80 | 1324.50 | 1329.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 1321.00 | 1324.50 | 1329.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1324.70 | 1325.26 | 1328.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1265.00 | 1325.26 | 1328.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 1257.61 | 1282.74 | 1297.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 15:15:00 | 1254.00 | 1276.59 | 1293.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 15:15:00 | 1254.86 | 1276.59 | 1293.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 15:15:00 | 1254.95 | 1276.59 | 1293.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 1272.40 | 1270.91 | 1286.02 | SL hit (close>ema200) qty=0.50 sl=1270.91 alert=retest2 |

### Cycle 122 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1269.60 | 1253.35 | 1252.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1275.20 | 1257.72 | 1254.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 1449.10 | 1455.76 | 1430.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 1449.10 | 1455.76 | 1430.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1446.00 | 1452.89 | 1433.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:00:00 | 1453.60 | 1453.03 | 1435.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 1452.50 | 1450.63 | 1438.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1452.50 | 1451.01 | 1439.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1462.30 | 1450.41 | 1440.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1475.50 | 1455.42 | 1443.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1495.30 | 1467.03 | 1464.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-02 11:15:00 | 1598.96 | 1568.49 | 1540.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 1739.50 | 1777.61 | 1778.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1676.00 | 1737.52 | 1756.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1662.00 | 1659.94 | 1686.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1662.00 | 1659.94 | 1686.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1683.00 | 1662.87 | 1674.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1683.00 | 1662.87 | 1674.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1684.90 | 1667.27 | 1675.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1687.00 | 1667.27 | 1675.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1708.00 | 1683.09 | 1681.49 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1676.00 | 1684.16 | 1684.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1674.00 | 1682.13 | 1683.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1658.10 | 1656.02 | 1667.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 1651.20 | 1656.02 | 1667.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1678.40 | 1660.49 | 1668.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1678.40 | 1660.49 | 1668.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1685.60 | 1665.51 | 1669.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1685.60 | 1665.51 | 1669.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1684.00 | 1672.97 | 1672.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 1687.00 | 1675.78 | 1673.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 1746.00 | 1756.14 | 1739.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 1746.00 | 1756.14 | 1739.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1742.00 | 1751.21 | 1740.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 1740.30 | 1751.21 | 1740.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1735.50 | 1748.07 | 1739.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 1735.50 | 1748.07 | 1739.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1739.50 | 1746.35 | 1739.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 1744.40 | 1744.07 | 1739.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 1778.50 | 1782.05 | 1782.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 1778.50 | 1782.05 | 1782.17 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1791.70 | 1783.98 | 1783.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 1807.90 | 1788.77 | 1785.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1782.70 | 1793.67 | 1789.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1782.70 | 1793.67 | 1789.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1782.70 | 1793.67 | 1789.85 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 1748.60 | 1780.77 | 1784.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 1725.00 | 1757.55 | 1767.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1765.60 | 1757.29 | 1763.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1765.60 | 1757.29 | 1763.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1765.60 | 1757.29 | 1763.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 1765.60 | 1757.29 | 1763.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1763.30 | 1758.49 | 1763.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 1758.90 | 1761.42 | 1763.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1769.90 | 1763.68 | 1764.28 | SL hit (close>static) qty=1.00 sl=1769.70 alert=retest2 |

### Cycle 130 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 1725.00 | 1708.60 | 1707.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 1727.00 | 1718.33 | 1712.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 1715.00 | 1717.66 | 1713.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 1715.00 | 1717.66 | 1713.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1711.80 | 1716.49 | 1713.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:15:00 | 1711.50 | 1716.49 | 1713.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1710.00 | 1715.19 | 1712.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:15:00 | 1708.80 | 1715.19 | 1712.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 1714.40 | 1715.03 | 1712.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:30:00 | 1713.60 | 1715.03 | 1712.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1713.20 | 1714.67 | 1712.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1713.30 | 1714.67 | 1712.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1696.40 | 1711.01 | 1711.43 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1721.60 | 1709.03 | 1708.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1728.80 | 1712.98 | 1710.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 1724.60 | 1725.89 | 1719.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:45:00 | 1724.10 | 1725.89 | 1719.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1717.00 | 1724.11 | 1719.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 1717.00 | 1724.11 | 1719.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1715.40 | 1722.37 | 1719.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1718.10 | 1722.37 | 1719.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1717.30 | 1721.35 | 1719.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1717.30 | 1721.35 | 1719.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1683.20 | 1712.64 | 1715.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 10:15:00 | 1677.20 | 1705.55 | 1711.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 1697.90 | 1697.57 | 1704.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 1686.50 | 1697.57 | 1704.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1526.10 | 1493.24 | 1502.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 1526.10 | 1493.24 | 1502.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1529.00 | 1500.39 | 1505.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:15:00 | 1531.40 | 1500.39 | 1505.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 1536.70 | 1513.74 | 1510.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 1553.50 | 1521.69 | 1514.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1540.50 | 1555.86 | 1543.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1540.50 | 1555.86 | 1543.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1540.50 | 1555.86 | 1543.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 1536.00 | 1555.86 | 1543.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1532.60 | 1551.21 | 1542.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1532.60 | 1551.21 | 1542.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1552.00 | 1551.37 | 1543.57 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1522.80 | 1540.11 | 1541.24 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1563.80 | 1544.97 | 1542.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 09:15:00 | 1570.00 | 1549.98 | 1544.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 1565.20 | 1569.01 | 1557.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 1565.20 | 1569.01 | 1557.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1563.10 | 1566.71 | 1558.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 1565.40 | 1566.71 | 1558.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1556.70 | 1564.26 | 1558.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 1553.50 | 1564.26 | 1558.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1564.20 | 1564.25 | 1559.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:45:00 | 1567.80 | 1564.15 | 1560.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 1553.40 | 1559.32 | 1559.19 | SL hit (close<static) qty=1.00 sl=1555.40 alert=retest2 |

### Cycle 137 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 1549.70 | 1557.40 | 1558.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 1542.50 | 1554.42 | 1556.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 1564.60 | 1554.31 | 1556.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1564.60 | 1554.31 | 1556.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1564.60 | 1554.31 | 1556.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 1564.60 | 1554.31 | 1556.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1560.90 | 1555.63 | 1556.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 1561.00 | 1555.63 | 1556.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1569.80 | 1559.88 | 1558.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1584.00 | 1565.15 | 1562.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 1574.10 | 1574.71 | 1568.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 1574.10 | 1574.71 | 1568.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1572.50 | 1574.78 | 1569.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1567.50 | 1574.78 | 1569.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1577.80 | 1579.51 | 1575.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 1585.40 | 1579.71 | 1577.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 1586.50 | 1583.10 | 1579.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 1570.90 | 1578.16 | 1578.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1570.90 | 1578.16 | 1578.21 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1586.60 | 1578.38 | 1578.21 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 1576.30 | 1577.86 | 1578.02 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 1581.70 | 1578.63 | 1578.36 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 1574.70 | 1577.84 | 1578.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1561.50 | 1574.12 | 1576.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1469.40 | 1445.85 | 1468.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1469.40 | 1445.85 | 1468.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1469.40 | 1445.85 | 1468.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1470.90 | 1445.85 | 1468.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1462.90 | 1449.26 | 1467.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 1457.60 | 1451.51 | 1467.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:45:00 | 1459.40 | 1454.41 | 1467.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1475.30 | 1460.60 | 1467.79 | SL hit (close>static) qty=1.00 sl=1471.20 alert=retest2 |

### Cycle 144 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1515.80 | 1480.28 | 1475.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 1535.20 | 1491.27 | 1481.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 1513.40 | 1513.95 | 1504.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 1509.90 | 1511.43 | 1505.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1509.90 | 1511.43 | 1505.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 1526.70 | 1510.79 | 1506.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 1530.90 | 1541.85 | 1543.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 1530.90 | 1541.85 | 1543.33 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1547.20 | 1542.96 | 1542.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 15:15:00 | 1550.00 | 1545.19 | 1544.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1550.00 | 1551.97 | 1549.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1550.00 | 1551.97 | 1549.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1550.00 | 1551.97 | 1549.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1558.20 | 1548.94 | 1548.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 1568.00 | 1553.42 | 1550.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1568.30 | 1578.72 | 1578.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1568.30 | 1578.72 | 1578.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1562.90 | 1573.13 | 1576.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1491.50 | 1482.36 | 1498.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 1491.50 | 1482.36 | 1498.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1489.00 | 1483.68 | 1497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 1489.00 | 1483.68 | 1497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1465.10 | 1464.55 | 1474.76 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 1486.00 | 1477.69 | 1476.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1490.90 | 1482.43 | 1479.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1552.30 | 1555.64 | 1534.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 1545.10 | 1555.64 | 1534.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1538.60 | 1546.02 | 1537.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 1538.50 | 1546.02 | 1537.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1537.80 | 1544.37 | 1537.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1532.60 | 1542.54 | 1537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1543.20 | 1542.67 | 1537.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 1553.30 | 1542.67 | 1537.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1611.70 | 1616.92 | 1617.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1611.70 | 1616.92 | 1617.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1608.00 | 1615.13 | 1616.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1618.40 | 1608.05 | 1610.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 1618.40 | 1608.05 | 1610.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1618.40 | 1608.05 | 1610.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1618.40 | 1608.05 | 1610.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1618.30 | 1610.10 | 1611.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 1619.40 | 1610.10 | 1611.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 1616.10 | 1612.50 | 1612.39 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 1610.50 | 1612.10 | 1612.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1597.20 | 1609.12 | 1610.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1611.40 | 1608.16 | 1610.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1611.40 | 1608.16 | 1610.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1611.40 | 1608.16 | 1610.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 1615.50 | 1608.16 | 1610.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1603.00 | 1607.13 | 1609.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 1599.10 | 1607.13 | 1609.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 1599.20 | 1606.17 | 1608.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1614.70 | 1599.68 | 1603.53 | SL hit (close>static) qty=1.00 sl=1611.40 alert=retest2 |

### Cycle 152 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1621.50 | 1607.61 | 1606.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1626.30 | 1613.13 | 1609.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 1626.70 | 1628.25 | 1620.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:00:00 | 1626.70 | 1628.25 | 1620.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1603.80 | 1624.13 | 1620.97 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1608.90 | 1617.86 | 1618.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 1598.50 | 1610.92 | 1613.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1602.80 | 1601.51 | 1608.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:00:00 | 1602.80 | 1601.51 | 1608.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1585.20 | 1594.27 | 1602.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1597.80 | 1594.27 | 1602.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1613.80 | 1598.18 | 1603.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 1613.80 | 1598.18 | 1603.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1598.10 | 1598.16 | 1602.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:30:00 | 1596.00 | 1598.71 | 1602.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1516.20 | 1535.64 | 1551.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1536.50 | 1535.64 | 1551.00 | SL hit (close>static) qty=0.50 sl=1535.64 alert=retest2 |

### Cycle 154 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 1579.00 | 1555.62 | 1555.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1630.00 | 1602.00 | 1589.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1633.20 | 1643.61 | 1627.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1633.20 | 1643.61 | 1627.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1626.90 | 1640.27 | 1627.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1626.90 | 1640.27 | 1627.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1624.80 | 1637.17 | 1627.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1641.30 | 1637.17 | 1627.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1635.50 | 1635.75 | 1628.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1619.20 | 1631.79 | 1627.81 | SL hit (close<static) qty=1.00 sl=1623.50 alert=retest2 |

### Cycle 155 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1606.80 | 1625.59 | 1626.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 1603.90 | 1614.74 | 1620.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 1627.00 | 1613.28 | 1617.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 1627.00 | 1613.28 | 1617.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1627.00 | 1613.28 | 1617.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 1622.00 | 1613.28 | 1617.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 1624.40 | 1615.50 | 1617.96 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1662.90 | 1626.78 | 1622.66 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1610.40 | 1629.34 | 1630.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1606.30 | 1620.26 | 1625.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1616.60 | 1590.52 | 1598.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1616.60 | 1590.52 | 1598.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1616.60 | 1590.52 | 1598.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1608.60 | 1590.52 | 1598.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1622.00 | 1596.81 | 1600.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1622.00 | 1596.81 | 1600.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1618.40 | 1605.27 | 1604.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1626.10 | 1616.06 | 1610.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1613.50 | 1616.13 | 1611.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 1613.50 | 1616.13 | 1611.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1613.90 | 1615.68 | 1611.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1609.50 | 1615.68 | 1611.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1624.00 | 1617.35 | 1612.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1626.50 | 1617.35 | 1612.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1636.00 | 1616.70 | 1615.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:15:00 | 1625.20 | 1619.71 | 1616.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1605.90 | 1615.07 | 1615.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 1605.90 | 1615.07 | 1615.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 1595.60 | 1609.04 | 1612.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 1546.70 | 1543.24 | 1557.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 1546.70 | 1543.24 | 1557.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1554.00 | 1546.96 | 1553.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1554.00 | 1546.96 | 1553.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1548.90 | 1547.35 | 1553.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 1558.20 | 1547.35 | 1553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1543.40 | 1546.56 | 1552.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 1536.10 | 1546.56 | 1552.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1459.29 | 1521.62 | 1536.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1521.80 | 1519.45 | 1532.62 | SL hit (close>ema200) qty=0.50 sl=1519.45 alert=retest2 |

### Cycle 160 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 1520.00 | 1512.23 | 1512.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1530.00 | 1515.78 | 1513.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1511.50 | 1520.61 | 1518.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1511.50 | 1520.61 | 1518.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1511.50 | 1520.61 | 1518.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1509.90 | 1520.61 | 1518.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1511.00 | 1518.69 | 1517.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 1511.40 | 1518.69 | 1517.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1517.40 | 1518.18 | 1517.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 1512.50 | 1518.18 | 1517.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1514.70 | 1517.48 | 1517.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 1514.70 | 1517.48 | 1517.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1518.00 | 1517.58 | 1517.29 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 1514.70 | 1517.01 | 1517.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1499.40 | 1513.49 | 1515.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1488.40 | 1484.60 | 1492.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1488.50 | 1484.60 | 1492.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1489.60 | 1485.60 | 1492.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 1489.00 | 1485.60 | 1492.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1487.70 | 1485.49 | 1491.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1487.70 | 1485.49 | 1491.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1489.50 | 1486.29 | 1490.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1498.40 | 1486.29 | 1490.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1490.80 | 1487.20 | 1490.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1497.50 | 1487.20 | 1490.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1486.30 | 1487.02 | 1490.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 1483.00 | 1487.02 | 1490.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1500.00 | 1491.02 | 1491.33 | SL hit (close>static) qty=1.00 sl=1494.20 alert=retest2 |

### Cycle 162 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1497.00 | 1492.21 | 1491.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1511.20 | 1496.01 | 1493.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 1513.10 | 1516.31 | 1510.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 1516.50 | 1516.31 | 1510.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1516.50 | 1516.35 | 1511.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 1519.80 | 1515.08 | 1511.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 1504.00 | 1512.86 | 1510.62 | SL hit (close<static) qty=1.00 sl=1508.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 1500.80 | 1508.13 | 1508.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1493.40 | 1503.40 | 1506.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1447.00 | 1443.55 | 1458.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 1448.50 | 1443.55 | 1458.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1447.20 | 1444.97 | 1451.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 1446.20 | 1444.97 | 1451.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1461.90 | 1448.18 | 1449.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 1461.90 | 1448.18 | 1449.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1462.90 | 1451.12 | 1451.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1466.50 | 1458.84 | 1455.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1458.50 | 1460.02 | 1456.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1458.50 | 1460.02 | 1456.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1458.50 | 1460.02 | 1456.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1457.00 | 1460.02 | 1456.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1458.50 | 1459.71 | 1456.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1458.50 | 1459.71 | 1456.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1464.30 | 1460.63 | 1457.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 1466.70 | 1460.63 | 1457.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 1469.60 | 1464.41 | 1460.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 1456.10 | 1459.96 | 1460.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1456.10 | 1459.96 | 1460.23 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1473.80 | 1462.73 | 1461.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 1476.00 | 1469.98 | 1465.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1466.80 | 1469.99 | 1466.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1466.80 | 1469.99 | 1466.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1466.80 | 1469.99 | 1466.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1474.30 | 1469.99 | 1466.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1450.90 | 1466.17 | 1465.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1450.90 | 1466.17 | 1465.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1446.90 | 1462.32 | 1463.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1437.80 | 1452.29 | 1458.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1419.40 | 1418.39 | 1428.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1428.00 | 1418.39 | 1428.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1421.40 | 1418.99 | 1427.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 1414.00 | 1418.03 | 1426.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 1413.30 | 1415.83 | 1424.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 1413.40 | 1415.52 | 1420.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 1414.60 | 1415.34 | 1420.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1418.10 | 1415.89 | 1420.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1418.10 | 1415.89 | 1420.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1424.40 | 1417.59 | 1420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:45:00 | 1423.80 | 1417.59 | 1420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1419.00 | 1417.87 | 1420.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1415.00 | 1417.87 | 1420.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1415.60 | 1417.42 | 1419.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1430.40 | 1419.63 | 1420.50 | SL hit (close>static) qty=1.00 sl=1425.40 alert=retest2 |

### Cycle 168 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 1445.90 | 1424.88 | 1422.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 1446.40 | 1429.19 | 1424.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 1422.50 | 1430.91 | 1428.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 10:15:00 | 1422.50 | 1430.91 | 1428.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1422.50 | 1430.91 | 1428.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1423.40 | 1430.91 | 1428.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1419.90 | 1428.71 | 1427.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 1419.90 | 1428.71 | 1427.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 1417.30 | 1426.43 | 1426.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1413.70 | 1423.88 | 1425.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1372.40 | 1347.46 | 1363.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1372.40 | 1347.46 | 1363.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1372.40 | 1347.46 | 1363.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1376.20 | 1347.46 | 1363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1365.00 | 1350.97 | 1363.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 1366.90 | 1350.97 | 1363.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1363.50 | 1353.48 | 1363.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 1359.00 | 1356.01 | 1363.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:30:00 | 1358.40 | 1356.39 | 1362.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 1357.90 | 1356.39 | 1362.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 1368.50 | 1359.07 | 1362.90 | SL hit (close>static) qty=1.00 sl=1367.30 alert=retest2 |

### Cycle 170 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1358.50 | 1343.28 | 1342.42 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 1319.00 | 1337.70 | 1340.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 1298.20 | 1319.68 | 1326.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1236.20 | 1234.90 | 1262.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1236.20 | 1234.90 | 1262.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1326.30 | 1254.02 | 1266.51 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1342.10 | 1286.53 | 1280.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1360.50 | 1333.27 | 1311.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 1348.60 | 1351.45 | 1332.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 1348.60 | 1351.45 | 1332.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1320.00 | 1347.65 | 1338.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1320.00 | 1347.65 | 1338.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1323.40 | 1342.80 | 1337.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1317.50 | 1342.80 | 1337.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1326.30 | 1333.55 | 1334.12 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1354.70 | 1336.99 | 1335.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1364.60 | 1342.51 | 1338.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1385.60 | 1390.85 | 1375.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 1385.60 | 1390.85 | 1375.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1379.70 | 1391.40 | 1383.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 1379.80 | 1391.40 | 1383.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1378.80 | 1388.88 | 1383.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 1377.20 | 1388.88 | 1383.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1370.20 | 1378.87 | 1379.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1331.80 | 1367.96 | 1374.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1347.70 | 1345.41 | 1357.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 1347.70 | 1345.41 | 1357.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1354.20 | 1348.30 | 1356.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 1354.20 | 1348.30 | 1356.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1353.90 | 1349.42 | 1356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 1353.20 | 1349.42 | 1356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 1353.60 | 1350.26 | 1355.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 1353.50 | 1350.26 | 1355.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1356.70 | 1351.55 | 1356.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1356.70 | 1351.55 | 1356.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1355.00 | 1352.24 | 1355.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1341.70 | 1352.24 | 1355.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 1345.40 | 1349.84 | 1354.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1348.00 | 1341.09 | 1347.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:15:00 | 1348.70 | 1343.49 | 1346.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1356.10 | 1346.01 | 1347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1356.10 | 1346.01 | 1347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 1356.40 | 1348.09 | 1348.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1356.40 | 1348.09 | 1348.06 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1343.20 | 1347.11 | 1347.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 1338.70 | 1345.43 | 1346.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1325.70 | 1325.21 | 1331.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1325.70 | 1325.21 | 1331.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1325.70 | 1325.21 | 1331.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1323.10 | 1326.35 | 1331.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1334.40 | 1332.32 | 1332.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 1334.40 | 1332.32 | 1332.30 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1331.40 | 1332.14 | 1332.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 1327.90 | 1331.29 | 1331.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1330.50 | 1328.83 | 1330.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1330.50 | 1328.83 | 1330.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1330.50 | 1328.83 | 1330.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 1327.10 | 1329.16 | 1330.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1260.74 | 1272.54 | 1288.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 11:15:00 | 1194.39 | 1219.54 | 1246.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 180 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1243.00 | 1223.84 | 1222.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1248.40 | 1232.03 | 1226.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1236.50 | 1238.99 | 1232.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 1239.50 | 1238.99 | 1232.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1235.50 | 1238.29 | 1232.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 1234.00 | 1238.29 | 1232.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1224.50 | 1235.53 | 1232.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1224.50 | 1235.53 | 1232.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1224.60 | 1233.35 | 1231.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 1229.50 | 1233.35 | 1231.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1217.60 | 1230.20 | 1230.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1217.60 | 1230.20 | 1230.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1220.00 | 1228.16 | 1229.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1197.60 | 1222.05 | 1226.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 1213.50 | 1213.22 | 1220.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:30:00 | 1214.20 | 1213.22 | 1220.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1184.50 | 1176.34 | 1188.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1188.90 | 1176.34 | 1188.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1196.70 | 1181.40 | 1188.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1196.70 | 1181.40 | 1188.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1192.30 | 1183.58 | 1189.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1188.40 | 1183.58 | 1189.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1190.00 | 1187.03 | 1189.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 1203.10 | 1191.69 | 1191.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1203.10 | 1191.69 | 1191.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1218.60 | 1197.07 | 1193.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1199.60 | 1219.71 | 1210.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1199.60 | 1219.71 | 1210.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1199.60 | 1219.71 | 1210.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1199.50 | 1219.71 | 1210.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1197.60 | 1215.29 | 1209.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1197.60 | 1215.29 | 1209.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1194.10 | 1204.33 | 1205.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1158.50 | 1188.39 | 1196.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1163.90 | 1152.54 | 1169.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1163.90 | 1152.54 | 1169.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1163.90 | 1152.54 | 1169.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 1171.90 | 1152.54 | 1169.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1167.30 | 1155.49 | 1169.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 1167.40 | 1155.49 | 1169.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1173.30 | 1159.05 | 1169.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 1174.00 | 1159.05 | 1169.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1186.40 | 1164.52 | 1171.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1185.00 | 1164.52 | 1171.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1186.70 | 1168.96 | 1172.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1186.50 | 1168.96 | 1172.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 1187.50 | 1174.99 | 1174.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1224.30 | 1184.86 | 1179.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1210.00 | 1210.10 | 1197.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 1196.00 | 1210.10 | 1197.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1183.00 | 1204.68 | 1196.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1183.00 | 1204.68 | 1196.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1176.20 | 1198.99 | 1194.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1176.20 | 1198.99 | 1194.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1177.80 | 1189.26 | 1190.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1173.00 | 1186.01 | 1188.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1189.60 | 1151.81 | 1163.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1189.60 | 1151.81 | 1163.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1189.60 | 1151.81 | 1163.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1189.60 | 1151.81 | 1163.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1178.50 | 1157.15 | 1165.28 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1190.60 | 1173.21 | 1171.38 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1144.20 | 1170.29 | 1170.77 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1187.90 | 1170.88 | 1169.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1192.00 | 1179.40 | 1174.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1265.50 | 1266.06 | 1238.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1265.50 | 1266.06 | 1238.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1283.20 | 1295.06 | 1279.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1292.00 | 1295.06 | 1279.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 1362.00 | 1365.03 | 1365.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 1362.00 | 1365.03 | 1365.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1344.00 | 1360.82 | 1363.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1324.70 | 1316.75 | 1324.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1324.70 | 1316.75 | 1324.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1324.70 | 1316.75 | 1324.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1329.80 | 1316.75 | 1324.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1322.60 | 1317.92 | 1323.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 1320.00 | 1323.45 | 1324.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 1320.50 | 1323.24 | 1324.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1332.00 | 1325.28 | 1324.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1332.00 | 1325.28 | 1324.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1339.80 | 1328.18 | 1326.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 1328.40 | 1328.44 | 1326.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 1328.40 | 1328.44 | 1326.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1316.80 | 1326.11 | 1325.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1316.80 | 1326.11 | 1325.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 1303.70 | 1321.63 | 1323.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 1277.80 | 1310.20 | 1318.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 11:15:00 | 1248.20 | 1244.47 | 1262.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 1248.20 | 1244.47 | 1262.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1248.20 | 1244.47 | 1262.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 1256.70 | 1244.47 | 1262.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1264.00 | 1252.76 | 1260.09 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1282.40 | 1263.84 | 1263.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1288.20 | 1268.71 | 1265.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 1269.20 | 1271.04 | 1268.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 1269.20 | 1271.04 | 1268.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1269.20 | 1271.04 | 1268.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 1269.20 | 1271.04 | 1268.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1270.00 | 1270.83 | 1268.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 1267.40 | 1270.83 | 1268.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1252.20 | 1267.11 | 1267.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 1252.20 | 1267.11 | 1267.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 1262.10 | 1266.11 | 1266.57 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-19 09:15:00 | 520.98 | 2023-06-20 09:15:00 | 516.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-06-22 09:15:00 | 532.28 | 2023-06-23 09:15:00 | 517.50 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2023-07-07 14:30:00 | 590.73 | 2023-07-13 15:15:00 | 598.50 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2023-07-10 09:15:00 | 597.50 | 2023-07-13 15:15:00 | 598.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2023-07-25 09:30:00 | 592.70 | 2023-07-27 09:15:00 | 608.65 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2023-07-25 11:15:00 | 593.05 | 2023-07-27 09:15:00 | 608.65 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2023-08-04 13:45:00 | 608.73 | 2023-08-08 14:15:00 | 616.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-08-04 15:00:00 | 608.75 | 2023-08-08 14:15:00 | 616.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-08-07 10:15:00 | 608.20 | 2023-08-08 14:15:00 | 616.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2023-08-08 10:45:00 | 608.00 | 2023-08-08 14:15:00 | 616.50 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-08-09 14:00:00 | 614.60 | 2023-08-10 09:15:00 | 608.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2023-08-17 14:45:00 | 584.00 | 2023-08-22 09:15:00 | 587.03 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-08-21 15:15:00 | 575.00 | 2023-08-22 09:15:00 | 587.03 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-09-01 15:15:00 | 569.50 | 2023-09-04 09:15:00 | 577.83 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-09-08 09:15:00 | 606.53 | 2023-09-11 09:15:00 | 667.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-10-05 10:15:00 | 658.15 | 2023-10-06 09:15:00 | 681.48 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2023-10-05 13:00:00 | 658.55 | 2023-10-06 09:15:00 | 681.48 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2023-10-10 11:30:00 | 660.55 | 2023-10-11 09:15:00 | 673.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-10-10 12:30:00 | 659.65 | 2023-10-11 09:15:00 | 673.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2023-10-10 14:30:00 | 660.58 | 2023-10-11 09:15:00 | 673.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-10-10 15:15:00 | 659.50 | 2023-10-11 09:15:00 | 673.50 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2023-10-20 11:15:00 | 674.68 | 2023-10-23 10:15:00 | 681.48 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-10-25 12:30:00 | 661.88 | 2023-10-27 13:15:00 | 664.28 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-10-26 09:15:00 | 649.80 | 2023-10-27 13:15:00 | 664.28 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-11-17 09:15:00 | 904.88 | 2023-11-17 14:15:00 | 888.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2023-11-22 14:00:00 | 856.95 | 2023-11-23 09:15:00 | 904.00 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest1 | 2023-12-01 09:15:00 | 958.50 | 2023-12-05 11:15:00 | 949.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest1 | 2023-12-01 15:00:00 | 950.45 | 2023-12-05 11:15:00 | 949.95 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2023-12-05 15:15:00 | 960.90 | 2023-12-06 09:15:00 | 952.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest1 | 2023-12-14 15:00:00 | 975.00 | 2023-12-15 09:15:00 | 960.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-12-18 13:45:00 | 952.78 | 2023-12-20 09:15:00 | 958.70 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-12-19 14:45:00 | 952.93 | 2023-12-20 09:15:00 | 958.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-01-09 09:15:00 | 943.35 | 2024-01-09 13:15:00 | 931.63 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-01-19 11:00:00 | 911.55 | 2024-01-23 14:15:00 | 865.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-20 09:30:00 | 911.33 | 2024-01-23 14:15:00 | 865.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-19 11:00:00 | 911.55 | 2024-01-24 14:15:00 | 881.95 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2024-01-20 09:30:00 | 911.33 | 2024-01-24 14:15:00 | 881.95 | STOP_HIT | 0.50 | 3.22% |
| BUY | retest2 | 2024-02-07 10:15:00 | 939.63 | 2024-02-12 10:15:00 | 948.60 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-02-14 09:30:00 | 932.43 | 2024-02-14 13:15:00 | 947.65 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-02-16 13:15:00 | 936.50 | 2024-02-23 13:15:00 | 931.50 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-02-16 15:15:00 | 937.00 | 2024-02-23 13:15:00 | 931.50 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-03-01 09:15:00 | 968.50 | 2024-03-04 10:15:00 | 953.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-03-01 10:00:00 | 967.80 | 2024-03-04 10:15:00 | 953.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-03-13 11:15:00 | 866.18 | 2024-03-20 09:15:00 | 822.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 09:30:00 | 866.03 | 2024-03-20 09:15:00 | 822.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 11:15:00 | 866.18 | 2024-03-21 09:15:00 | 865.63 | STOP_HIT | 0.50 | 0.06% |
| SELL | retest2 | 2024-03-15 09:30:00 | 866.03 | 2024-03-21 09:15:00 | 865.63 | STOP_HIT | 0.50 | 0.05% |
| BUY | retest2 | 2024-04-09 11:15:00 | 935.73 | 2024-04-09 15:15:00 | 932.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-04-09 14:45:00 | 936.83 | 2024-04-09 15:15:00 | 932.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-04-12 09:15:00 | 963.70 | 2024-04-19 10:15:00 | 968.83 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2024-04-25 11:30:00 | 1065.00 | 2024-05-02 09:15:00 | 1040.68 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-04-26 12:00:00 | 1060.58 | 2024-05-02 09:15:00 | 1040.68 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-04-29 12:30:00 | 1060.50 | 2024-05-02 09:15:00 | 1040.68 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-04-30 09:15:00 | 1061.00 | 2024-05-02 09:15:00 | 1040.68 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-05-08 10:00:00 | 1078.72 | 2024-05-09 12:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-05-08 11:30:00 | 1074.10 | 2024-05-09 12:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-05-08 12:00:00 | 1073.88 | 2024-05-09 12:15:00 | 1052.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-05-08 12:30:00 | 1074.22 | 2024-05-09 12:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-06-11 12:15:00 | 1053.50 | 2024-06-18 11:15:00 | 1043.08 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1051.30 | 2024-06-18 11:15:00 | 1043.08 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-12 13:15:00 | 1049.20 | 2024-06-18 11:15:00 | 1043.08 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-06-12 13:45:00 | 1049.47 | 2024-06-18 11:15:00 | 1043.08 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1050.83 | 2024-06-18 11:15:00 | 1043.08 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-06-25 11:00:00 | 1013.18 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.56% |
| SELL | retest2 | 2024-06-25 13:00:00 | 1015.23 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2024-06-25 13:45:00 | 1015.00 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.36% |
| SELL | retest2 | 2024-06-26 09:15:00 | 1015.10 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.35% |
| SELL | retest2 | 2024-06-26 11:30:00 | 1013.50 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.52% |
| SELL | retest2 | 2024-06-26 12:00:00 | 1013.65 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.51% |
| SELL | retest2 | 2024-06-26 12:45:00 | 1012.75 | 2024-06-28 09:15:00 | 1089.75 | STOP_HIT | 1.00 | -7.60% |
| BUY | retest1 | 2024-07-02 14:00:00 | 1209.97 | 2024-07-03 09:15:00 | 1162.03 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-07-08 09:30:00 | 1154.60 | 2024-07-09 09:15:00 | 1167.05 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-08 12:00:00 | 1153.55 | 2024-07-09 09:15:00 | 1167.05 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-07-08 12:30:00 | 1155.90 | 2024-07-09 09:15:00 | 1167.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-08 13:15:00 | 1155.88 | 2024-07-09 09:15:00 | 1167.05 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-07-09 13:45:00 | 1154.68 | 2024-07-11 09:15:00 | 1179.97 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-07-09 14:15:00 | 1155.40 | 2024-07-11 09:15:00 | 1179.97 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-10 10:00:00 | 1146.20 | 2024-07-11 09:15:00 | 1179.97 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-07-15 15:15:00 | 1206.50 | 2024-07-16 09:15:00 | 1198.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-25 12:00:00 | 1203.43 | 2024-08-01 12:15:00 | 1221.45 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1186.05 | 2024-08-08 10:15:00 | 1235.05 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1275.00 | 2024-08-16 13:15:00 | 1391.15 | TARGET_HIT | 1.00 | 9.11% |
| BUY | retest2 | 2024-08-14 09:45:00 | 1273.68 | 2024-08-16 13:15:00 | 1391.19 | TARGET_HIT | 1.00 | 9.23% |
| BUY | retest2 | 2024-08-14 11:45:00 | 1264.68 | 2024-08-16 15:15:00 | 1402.50 | TARGET_HIT | 1.00 | 10.90% |
| BUY | retest2 | 2024-08-14 14:00:00 | 1264.72 | 2024-08-16 15:15:00 | 1401.05 | TARGET_HIT | 1.00 | 10.78% |
| BUY | retest2 | 2024-08-27 11:45:00 | 1526.95 | 2024-08-28 10:15:00 | 1502.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-08-27 15:00:00 | 1537.35 | 2024-08-28 10:15:00 | 1502.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-09-04 11:15:00 | 1424.65 | 2024-09-10 11:15:00 | 1353.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 12:30:00 | 1425.25 | 2024-09-10 11:15:00 | 1353.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 09:30:00 | 1424.00 | 2024-09-10 11:15:00 | 1352.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 10:15:00 | 1424.50 | 2024-09-10 11:15:00 | 1353.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 14:30:00 | 1419.85 | 2024-09-10 12:15:00 | 1348.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 11:15:00 | 1424.65 | 2024-09-11 09:15:00 | 1363.60 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2024-09-04 12:30:00 | 1425.25 | 2024-09-11 09:15:00 | 1363.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2024-09-05 09:30:00 | 1424.00 | 2024-09-11 09:15:00 | 1363.60 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-09-05 10:15:00 | 1424.50 | 2024-09-11 09:15:00 | 1363.60 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2024-09-05 14:30:00 | 1419.85 | 2024-09-11 09:15:00 | 1363.60 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2024-09-06 10:00:00 | 1413.00 | 2024-09-12 15:15:00 | 1382.00 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2024-09-25 11:30:00 | 1502.05 | 2024-10-01 10:15:00 | 1426.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 11:30:00 | 1502.05 | 2024-10-01 15:15:00 | 1441.10 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest1 | 2024-10-14 09:15:00 | 1507.10 | 2024-10-14 12:15:00 | 1582.45 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-14 09:15:00 | 1507.10 | 2024-10-16 11:15:00 | 1576.30 | STOP_HIT | 0.50 | 4.59% |
| BUY | retest2 | 2024-10-17 09:15:00 | 1592.50 | 2024-10-17 13:15:00 | 1548.35 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-10-17 09:45:00 | 1576.20 | 2024-10-17 13:15:00 | 1548.35 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-11-04 13:00:00 | 1526.80 | 2024-11-08 10:15:00 | 1676.40 | TARGET_HIT | 1.00 | 9.80% |
| BUY | retest2 | 2024-11-05 09:15:00 | 1524.00 | 2024-11-08 10:15:00 | 1676.90 | TARGET_HIT | 1.00 | 10.03% |
| BUY | retest2 | 2024-11-05 10:00:00 | 1524.45 | 2024-11-08 10:15:00 | 1673.43 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2024-11-05 12:30:00 | 1521.30 | 2024-11-11 12:15:00 | 1563.00 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1587.25 | 2024-11-11 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-11-12 10:15:00 | 1558.35 | 2024-11-13 09:15:00 | 1480.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:15:00 | 1558.35 | 2024-11-14 09:15:00 | 1492.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest1 | 2024-11-25 09:15:00 | 1563.40 | 2024-11-28 09:15:00 | 1641.57 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-25 09:15:00 | 1563.40 | 2024-11-28 12:15:00 | 1599.15 | STOP_HIT | 0.50 | 2.29% |
| BUY | retest2 | 2024-11-29 09:15:00 | 1619.00 | 2024-12-05 09:15:00 | 1780.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-27 10:30:00 | 1777.20 | 2025-01-01 11:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-12-27 11:45:00 | 1774.00 | 2025-01-01 11:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-27 14:45:00 | 1776.70 | 2025-01-01 11:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-12-30 10:45:00 | 1778.65 | 2025-01-01 11:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-09 12:15:00 | 1696.00 | 2025-01-10 13:15:00 | 1611.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 1696.00 | 2025-01-13 14:15:00 | 1526.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-17 09:45:00 | 1610.90 | 2025-01-17 12:15:00 | 1601.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1583.00 | 2025-01-22 09:15:00 | 1503.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1583.00 | 2025-01-23 09:15:00 | 1515.30 | STOP_HIT | 0.50 | 4.28% |
| BUY | retest2 | 2025-02-07 11:00:00 | 1354.00 | 2025-02-10 09:15:00 | 1303.00 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1244.15 | 2025-02-17 09:15:00 | 1181.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1244.15 | 2025-02-17 13:15:00 | 1203.80 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-03-12 11:15:00 | 1086.00 | 2025-03-18 11:15:00 | 1112.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-12 13:30:00 | 1084.80 | 2025-03-18 11:15:00 | 1112.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-03-12 14:30:00 | 1087.40 | 2025-03-18 11:15:00 | 1112.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1073.00 | 2025-03-18 11:15:00 | 1112.00 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1161.55 | 2025-04-11 14:15:00 | 1168.35 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-04-17 12:15:00 | 1247.50 | 2025-04-21 15:15:00 | 1309.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-17 13:30:00 | 1244.90 | 2025-04-21 15:15:00 | 1307.15 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-17 14:15:00 | 1243.90 | 2025-04-21 15:15:00 | 1306.10 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-17 12:15:00 | 1247.50 | 2025-04-23 09:15:00 | 1308.10 | STOP_HIT | 0.50 | 4.86% |
| BUY | retest1 | 2025-04-17 13:30:00 | 1244.90 | 2025-04-23 09:15:00 | 1308.10 | STOP_HIT | 0.50 | 5.08% |
| BUY | retest1 | 2025-04-17 14:15:00 | 1243.90 | 2025-04-23 09:15:00 | 1308.10 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest1 | 2025-04-21 09:15:00 | 1276.60 | 2025-04-23 14:15:00 | 1340.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 09:15:00 | 1276.60 | 2025-04-25 09:15:00 | 1321.10 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2025-04-28 11:15:00 | 1343.90 | 2025-04-30 10:15:00 | 1322.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-04-28 12:30:00 | 1344.20 | 2025-04-30 10:15:00 | 1322.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-04-29 09:15:00 | 1368.20 | 2025-04-30 10:15:00 | 1322.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-05-02 11:15:00 | 1320.00 | 2025-05-06 14:15:00 | 1257.61 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-05-02 12:00:00 | 1320.90 | 2025-05-06 15:15:00 | 1254.00 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-05-02 13:30:00 | 1323.80 | 2025-05-06 15:15:00 | 1254.86 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1321.00 | 2025-05-06 15:15:00 | 1254.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 1320.00 | 2025-05-07 11:15:00 | 1272.40 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-05-02 12:00:00 | 1320.90 | 2025-05-07 11:15:00 | 1272.40 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-05-02 13:30:00 | 1323.80 | 2025-05-07 11:15:00 | 1272.40 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1321.00 | 2025-05-07 11:15:00 | 1272.40 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-05-05 09:15:00 | 1265.00 | 2025-05-09 09:15:00 | 1201.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 09:15:00 | 1265.00 | 2025-05-09 15:15:00 | 1231.00 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2025-05-21 11:00:00 | 1453.60 | 2025-06-02 11:15:00 | 1598.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 14:00:00 | 1452.50 | 2025-06-02 11:15:00 | 1597.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1452.50 | 2025-06-02 11:15:00 | 1597.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:15:00 | 1462.30 | 2025-06-02 11:15:00 | 1608.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1495.30 | 2025-06-02 11:15:00 | 1644.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 14:15:00 | 1744.40 | 2025-07-02 15:15:00 | 1778.50 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-07-09 14:15:00 | 1758.90 | 2025-07-10 09:15:00 | 1769.90 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-07-10 11:30:00 | 1761.50 | 2025-07-14 13:15:00 | 1673.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 12:30:00 | 1761.20 | 2025-07-14 13:15:00 | 1673.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 11:30:00 | 1761.50 | 2025-07-15 11:15:00 | 1686.90 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-07-10 12:30:00 | 1761.20 | 2025-07-15 11:15:00 | 1686.90 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest2 | 2025-08-11 14:45:00 | 1567.80 | 2025-08-12 12:15:00 | 1553.40 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-20 14:45:00 | 1585.40 | 2025-08-21 14:15:00 | 1570.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-08-21 09:45:00 | 1586.50 | 2025-08-21 14:15:00 | 1570.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-01 11:45:00 | 1457.60 | 2025-09-01 14:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-01 12:45:00 | 1459.40 | 2025-09-01 14:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-09-05 09:15:00 | 1526.70 | 2025-09-11 13:15:00 | 1530.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1558.20 | 2025-09-22 11:15:00 | 1568.30 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-09-17 11:00:00 | 1568.00 | 2025-09-22 11:15:00 | 1568.30 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-10-09 11:15:00 | 1553.30 | 2025-10-20 09:15:00 | 1611.70 | STOP_HIT | 1.00 | 3.76% |
| SELL | retest2 | 2025-10-24 11:15:00 | 1599.10 | 2025-10-27 09:15:00 | 1614.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-24 12:15:00 | 1599.20 | 2025-10-27 09:15:00 | 1614.70 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-03 14:30:00 | 1596.00 | 2025-11-07 09:15:00 | 1516.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:30:00 | 1596.00 | 2025-11-07 09:15:00 | 1536.50 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest2 | 2025-11-14 09:15:00 | 1641.30 | 2025-11-14 12:15:00 | 1619.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1635.50 | 2025-11-14 12:15:00 | 1619.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-17 09:15:00 | 1633.40 | 2025-11-18 09:15:00 | 1606.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-17 13:15:00 | 1639.20 | 2025-11-18 09:15:00 | 1606.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1626.50 | 2025-12-01 14:15:00 | 1605.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1636.00 | 2025-12-01 14:15:00 | 1605.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-01 11:15:00 | 1625.20 | 2025-12-01 14:15:00 | 1605.90 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-08 11:15:00 | 1536.10 | 2025-12-09 09:15:00 | 1459.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 11:15:00 | 1536.10 | 2025-12-09 11:15:00 | 1521.80 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2025-12-19 11:15:00 | 1483.00 | 2025-12-19 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-24 10:30:00 | 1519.80 | 2025-12-24 11:15:00 | 1504.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-05 12:15:00 | 1466.70 | 2026-01-06 15:15:00 | 1456.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-01-05 14:45:00 | 1469.60 | 2026-01-06 15:15:00 | 1456.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-13 10:45:00 | 1414.00 | 2026-01-16 09:15:00 | 1430.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1413.30 | 2026-01-16 09:15:00 | 1430.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-14 10:15:00 | 1413.40 | 2026-01-16 10:15:00 | 1445.90 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-01-14 11:00:00 | 1414.60 | 2026-01-16 10:15:00 | 1445.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1415.00 | 2026-01-16 10:15:00 | 1445.90 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1415.60 | 2026-01-16 10:15:00 | 1445.90 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-01-22 14:00:00 | 1359.00 | 2026-01-23 09:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-22 14:30:00 | 1358.40 | 2026-01-23 09:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-22 15:00:00 | 1357.90 | 2026-01-23 09:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-23 11:30:00 | 1357.30 | 2026-01-28 14:15:00 | 1358.50 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1341.70 | 2026-02-18 15:15:00 | 1356.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1345.40 | 2026-02-18 15:15:00 | 1356.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-18 09:30:00 | 1348.00 | 2026-02-18 15:15:00 | 1356.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-02-18 14:15:00 | 1348.70 | 2026-02-18 15:15:00 | 1356.40 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-23 10:30:00 | 1323.10 | 2026-02-24 10:15:00 | 1334.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1327.10 | 2026-03-02 09:15:00 | 1260.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1327.10 | 2026-03-04 11:15:00 | 1194.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1188.40 | 2026-03-17 15:15:00 | 1203.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-17 14:15:00 | 1190.00 | 2026-03-17 15:15:00 | 1203.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1292.00 | 2026-04-21 15:15:00 | 1362.00 | STOP_HIT | 1.00 | 5.42% |
| SELL | retest2 | 2026-04-28 09:30:00 | 1320.00 | 2026-04-29 09:15:00 | 1332.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-28 14:00:00 | 1320.50 | 2026-04-29 09:15:00 | 1332.00 | STOP_HIT | 1.00 | -0.87% |
