# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 561.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 120 |
| ALERT1 | 90 |
| ALERT2 | 85 |
| ALERT2_SKIP | 39 |
| ALERT3 | 217 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 96 |
| PARTIAL | 20 |
| TARGET_HIT | 10 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 64 / 57
- **Target hits / Stop hits / Partials:** 10 / 91 / 20
- **Avg / median % per leg:** 1.27% / 0.59%
- **Sum % (uncompounded):** 154.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 27 | 54.0% | 6 | 42 | 2 | 0.96% | 47.8% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.38% | 13.5% |
| BUY @ 3rd Alert (retest2) | 46 | 23 | 50.0% | 6 | 40 | 0 | 0.74% | 34.3% |
| SELL (all) | 71 | 37 | 52.1% | 4 | 49 | 18 | 1.50% | 106.3% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.43% | 26.6% |
| SELL @ 3rd Alert (retest2) | 65 | 31 | 47.7% | 4 | 46 | 15 | 1.23% | 79.7% |
| retest1 (combined) | 10 | 10 | 100.0% | 0 | 5 | 5 | 4.01% | 40.1% |
| retest2 (combined) | 111 | 54 | 48.6% | 10 | 86 | 15 | 1.03% | 113.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 383.45 | 384.74 | 384.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 382.10 | 384.21 | 384.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 12:15:00 | 382.40 | 382.13 | 383.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 13:00:00 | 382.40 | 382.13 | 383.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 383.60 | 382.42 | 383.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 383.45 | 382.42 | 383.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 382.10 | 382.36 | 382.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 381.40 | 382.36 | 382.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 384.30 | 376.36 | 375.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 384.30 | 376.36 | 375.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 386.80 | 378.45 | 376.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 377.20 | 392.31 | 387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 351.90 | 384.23 | 383.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 351.90 | 384.23 | 383.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 334.90 | 374.36 | 379.46 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 368.75 | 363.22 | 363.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 374.55 | 365.49 | 364.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 421.40 | 423.66 | 414.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 421.40 | 423.66 | 414.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 415.50 | 419.94 | 415.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 423.20 | 416.46 | 414.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 439.15 | 442.23 | 442.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 439.15 | 442.23 | 442.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 439.00 | 441.59 | 442.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 441.25 | 440.22 | 441.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 442.00 | 440.57 | 441.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 442.75 | 440.57 | 441.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 441.65 | 440.79 | 441.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 439.10 | 441.02 | 441.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 439.05 | 440.86 | 441.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:45:00 | 439.70 | 434.92 | 436.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 442.40 | 437.24 | 437.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 442.40 | 437.24 | 437.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 444.45 | 438.68 | 437.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 482.65 | 483.88 | 473.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 09:45:00 | 481.50 | 483.88 | 473.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 522.20 | 530.38 | 521.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 522.90 | 530.38 | 521.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 526.65 | 529.64 | 522.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 528.50 | 528.91 | 522.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 514.55 | 525.97 | 522.18 | SL hit (close<static) qty=1.00 sl=521.60 alert=retest2 |

### Cycle 7 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 512.00 | 519.68 | 519.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 505.10 | 516.77 | 518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 491.40 | 491.17 | 496.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:30:00 | 490.15 | 491.17 | 496.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 472.10 | 478.93 | 484.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 478.50 | 478.93 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 487.40 | 478.47 | 482.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:45:00 | 483.90 | 478.47 | 482.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 489.85 | 480.74 | 483.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 494.65 | 480.74 | 483.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 483.00 | 481.08 | 483.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 513.80 | 481.08 | 483.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 530.55 | 490.97 | 487.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 541.50 | 524.74 | 519.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 551.25 | 551.32 | 542.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:30:00 | 552.05 | 551.32 | 542.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 552.25 | 553.78 | 550.39 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 539.90 | 547.00 | 547.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 532.55 | 542.91 | 545.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 517.75 | 509.90 | 519.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 506.75 | 505.40 | 512.15 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 529.65 | 512.69 | 512.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 516.45 | 519.59 | 519.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 514.00 | 518.47 | 519.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 510.00 | 515.16 | 517.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 14:15:00 | 519.75 | 516.11 | 515.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 519.75 | 516.11 | 515.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 535.45 | 520.28 | 517.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 592.10 | 592.56 | 574.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 592.10 | 592.56 | 574.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 593.40 | 594.98 | 589.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:30:00 | 595.05 | 594.49 | 589.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 586.15 | 592.82 | 589.52 | SL hit (close<static) qty=1.00 sl=589.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 584.45 | 588.35 | 588.58 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 632.40 | 596.45 | 592.08 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 595.80 | 605.45 | 605.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 15:15:00 | 593.10 | 596.39 | 598.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 11:15:00 | 597.70 | 596.39 | 598.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 11:30:00 | 597.00 | 596.39 | 598.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 599.05 | 596.92 | 598.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 599.05 | 596.92 | 598.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 602.95 | 598.13 | 598.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 602.95 | 598.13 | 598.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 603.80 | 599.26 | 599.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 612.80 | 602.75 | 600.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 602.60 | 602.72 | 600.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:00:00 | 602.60 | 602.72 | 600.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 599.80 | 602.14 | 600.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:00:00 | 599.80 | 602.14 | 600.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 601.85 | 602.08 | 600.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:30:00 | 601.80 | 602.08 | 600.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 606.55 | 602.97 | 601.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:30:00 | 608.00 | 604.60 | 602.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 600.80 | 610.71 | 611.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 600.80 | 610.71 | 611.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 598.55 | 606.57 | 609.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 609.80 | 606.34 | 608.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 611.20 | 607.31 | 609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 619.55 | 607.31 | 609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 618.00 | 609.45 | 609.88 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 613.35 | 610.23 | 610.20 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 11:15:00 | 607.75 | 609.73 | 609.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 10:15:00 | 606.60 | 608.68 | 609.31 | Break + close below crossover candle low |

### Cycle 20 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 624.95 | 611.68 | 610.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 636.90 | 620.15 | 615.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 644.05 | 644.62 | 637.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 12:00:00 | 644.05 | 644.62 | 637.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 676.80 | 654.22 | 647.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 680.00 | 664.25 | 658.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-20 13:15:00 | 748.00 | 709.67 | 687.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 709.70 | 717.69 | 718.14 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 729.10 | 719.75 | 718.77 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 714.50 | 718.71 | 718.75 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 722.00 | 719.34 | 719.03 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 714.70 | 718.41 | 718.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 700.80 | 714.89 | 717.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 717.80 | 709.72 | 712.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 721.35 | 712.05 | 713.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 717.25 | 712.05 | 713.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 715.90 | 712.82 | 713.41 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 719.30 | 714.11 | 713.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 731.00 | 718.43 | 715.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 716.75 | 719.48 | 717.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 707.75 | 717.14 | 716.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 707.75 | 717.14 | 716.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 704.95 | 714.70 | 715.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 702.65 | 712.29 | 714.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 723.00 | 710.38 | 711.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 722.00 | 712.70 | 712.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:45:00 | 724.25 | 712.70 | 712.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 13:15:00 | 719.85 | 714.13 | 713.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 15:15:00 | 727.00 | 717.44 | 715.13 | Break + close above crossover candle high |

### Cycle 29 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 690.85 | 712.78 | 713.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 671.85 | 695.49 | 704.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 685.85 | 684.83 | 696.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 686.65 | 684.83 | 696.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 702.00 | 688.26 | 697.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 702.00 | 688.26 | 697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 707.55 | 692.12 | 698.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 715.00 | 692.12 | 698.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 724.00 | 705.41 | 703.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 735.00 | 711.32 | 706.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 745.40 | 748.10 | 738.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 13:15:00 | 751.75 | 747.63 | 739.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 13:45:00 | 754.20 | 748.39 | 740.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 750.50 | 749.53 | 743.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 756.85 | 751.08 | 744.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:15:00 | 789.34 | 771.25 | 762.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:15:00 | 791.91 | 771.25 | 762.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 766.25 | 770.57 | 763.71 | SL hit (close<ema200) qty=0.50 sl=770.57 alert=retest1 |

### Cycle 31 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 752.90 | 765.05 | 765.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 731.00 | 745.86 | 754.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 722.30 | 713.47 | 729.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 722.30 | 713.47 | 729.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 676.50 | 667.14 | 679.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:15:00 | 681.00 | 667.14 | 679.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 681.40 | 670.00 | 679.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 681.40 | 670.00 | 679.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 681.00 | 672.20 | 679.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:30:00 | 682.15 | 672.20 | 679.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 680.90 | 673.94 | 679.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:15:00 | 685.00 | 673.94 | 679.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 685.00 | 676.15 | 680.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 680.95 | 676.15 | 680.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 680.00 | 676.92 | 680.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 671.80 | 675.38 | 679.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 671.50 | 675.90 | 678.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:30:00 | 674.60 | 676.14 | 678.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:15:00 | 673.40 | 676.14 | 678.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 679.10 | 676.73 | 678.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 679.10 | 676.73 | 678.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 684.55 | 678.30 | 679.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 684.55 | 678.30 | 679.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 685.35 | 679.71 | 679.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 696.50 | 679.71 | 679.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 688.15 | 681.40 | 680.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 688.15 | 681.40 | 680.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 728.00 | 698.23 | 690.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 722.05 | 735.61 | 719.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 719.95 | 732.48 | 719.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 719.95 | 732.48 | 719.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 723.05 | 730.59 | 720.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 727.95 | 729.48 | 721.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:45:00 | 731.35 | 729.38 | 721.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 728.10 | 728.30 | 722.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 729.40 | 729.64 | 723.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 749.75 | 754.63 | 748.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 749.75 | 754.63 | 748.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 750.00 | 753.70 | 748.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 747.95 | 753.70 | 748.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 748.95 | 752.75 | 748.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 748.50 | 752.75 | 748.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 750.35 | 752.27 | 748.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 14:15:00 | 740.40 | 746.08 | 746.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 740.40 | 746.08 | 746.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 733.00 | 743.47 | 745.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 745.65 | 743.44 | 745.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 743.50 | 743.45 | 744.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:30:00 | 738.30 | 741.97 | 744.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 701.38 | 721.78 | 731.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-14 13:15:00 | 664.47 | 681.18 | 698.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 675.80 | 671.90 | 671.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 677.10 | 673.54 | 672.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 673.05 | 676.42 | 674.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 671.80 | 675.50 | 674.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 671.80 | 675.50 | 674.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 673.00 | 674.04 | 673.88 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 671.65 | 673.56 | 673.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 662.00 | 671.25 | 672.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:30:00 | 671.95 | 665.38 | 667.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 668.25 | 665.95 | 667.96 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 674.00 | 669.56 | 669.20 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 663.70 | 668.62 | 668.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 662.55 | 667.13 | 668.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 666.80 | 666.36 | 667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 667.70 | 666.63 | 667.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 664.00 | 666.18 | 667.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 664.40 | 666.18 | 667.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 663.65 | 665.40 | 666.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 677.80 | 668.33 | 667.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 677.80 | 668.33 | 667.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 701.55 | 675.48 | 671.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 13:15:00 | 723.15 | 725.27 | 717.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:45:00 | 719.40 | 725.27 | 717.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 736.45 | 738.36 | 735.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 730.00 | 738.36 | 735.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 730.00 | 736.69 | 735.17 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 729.00 | 733.24 | 733.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 727.25 | 731.19 | 732.72 | Break + close below crossover candle low |

### Cycle 40 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 746.00 | 733.58 | 733.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 11:15:00 | 751.70 | 739.08 | 736.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 739.30 | 746.46 | 741.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 739.70 | 745.11 | 741.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 736.65 | 745.11 | 741.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 748.00 | 745.69 | 742.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:45:00 | 751.00 | 746.56 | 743.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:30:00 | 751.50 | 749.19 | 744.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-17 09:15:00 | 826.10 | 790.50 | 771.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 12:15:00 | 821.25 | 828.55 | 829.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 814.65 | 822.14 | 825.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 818.70 | 815.99 | 821.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 818.70 | 815.99 | 821.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 823.00 | 817.39 | 821.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 830.40 | 818.34 | 821.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 820.45 | 818.76 | 821.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 822.70 | 818.76 | 821.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 829.50 | 820.85 | 821.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 829.50 | 820.85 | 821.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 823.45 | 821.37 | 821.75 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 826.95 | 822.49 | 822.22 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 818.75 | 822.03 | 822.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 815.25 | 820.31 | 821.32 | Break + close below crossover candle low |

### Cycle 44 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 841.50 | 824.55 | 823.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 15:15:00 | 850.00 | 829.64 | 825.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 829.55 | 829.62 | 825.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:00:00 | 829.55 | 829.62 | 825.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 871.80 | 864.96 | 856.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:30:00 | 858.00 | 864.96 | 856.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 865.95 | 872.52 | 866.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:15:00 | 857.70 | 872.52 | 866.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 854.85 | 868.98 | 865.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:45:00 | 855.35 | 868.98 | 865.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 860.20 | 867.23 | 865.03 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 854.10 | 862.97 | 863.37 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 875.00 | 863.28 | 863.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 881.95 | 869.61 | 866.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 10:15:00 | 914.10 | 916.78 | 900.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:45:00 | 913.10 | 916.78 | 900.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 904.35 | 912.69 | 901.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 907.85 | 912.69 | 901.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 900.80 | 910.32 | 901.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 900.80 | 910.32 | 901.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 907.85 | 909.82 | 902.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 910.70 | 908.46 | 902.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 882.75 | 903.32 | 900.38 | SL hit (close<static) qty=1.00 sl=896.50 alert=retest2 |

### Cycle 47 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 887.15 | 899.33 | 899.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 15:15:00 | 885.00 | 894.61 | 897.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 849.50 | 835.55 | 847.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 857.40 | 839.92 | 848.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 857.40 | 839.92 | 848.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 877.00 | 854.07 | 854.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 896.00 | 862.46 | 857.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 884.95 | 885.29 | 876.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 884.95 | 885.29 | 876.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 894.25 | 904.77 | 896.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 894.25 | 904.77 | 896.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 877.20 | 899.26 | 894.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 877.20 | 899.26 | 894.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 876.80 | 894.76 | 892.91 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 868.95 | 889.60 | 890.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 842.60 | 871.71 | 881.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 868.45 | 847.74 | 860.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 866.50 | 851.49 | 861.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 859.10 | 853.47 | 861.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 816.14 | 833.40 | 845.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 773.19 | 811.29 | 832.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 616.75 | 600.62 | 600.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 627.35 | 609.26 | 604.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 601.10 | 615.59 | 609.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 583.25 | 609.12 | 606.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 583.25 | 609.12 | 606.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 581.15 | 603.52 | 604.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 573.40 | 585.63 | 593.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 594.05 | 582.62 | 589.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 587.65 | 583.62 | 589.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 587.50 | 583.62 | 589.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 585.75 | 584.23 | 587.87 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 613.45 | 590.04 | 589.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 14:15:00 | 620.25 | 606.86 | 600.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 586.90 | 619.71 | 613.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 590.15 | 613.80 | 611.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:45:00 | 591.20 | 613.80 | 611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 11:15:00 | 590.00 | 609.04 | 609.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 561.45 | 591.06 | 599.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 597.20 | 585.21 | 591.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 599.85 | 588.14 | 592.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 605.20 | 588.14 | 592.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 597.85 | 595.04 | 594.71 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 581.75 | 592.53 | 593.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 571.35 | 588.29 | 591.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 541.60 | 518.20 | 531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 540.30 | 522.62 | 532.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 546.80 | 522.62 | 532.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 539.00 | 534.51 | 535.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 534.90 | 534.51 | 535.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:45:00 | 532.80 | 533.01 | 534.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:45:00 | 537.15 | 534.92 | 535.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 12:15:00 | 536.70 | 534.92 | 535.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 542.90 | 536.87 | 536.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 542.90 | 536.87 | 536.05 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 523.45 | 533.59 | 534.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 515.75 | 527.38 | 531.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 520.35 | 516.06 | 521.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 457.65 | 450.86 | 462.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 469.10 | 450.86 | 462.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 449.80 | 448.43 | 456.76 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 463.00 | 459.80 | 459.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 468.60 | 462.86 | 461.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 465.00 | 465.35 | 463.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 465.00 | 465.35 | 463.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 464.00 | 465.08 | 463.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 474.20 | 465.08 | 463.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 470.25 | 466.11 | 463.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:00:00 | 487.00 | 470.93 | 466.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 483.75 | 486.35 | 482.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 12:15:00 | 489.85 | 493.72 | 493.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 489.85 | 493.72 | 493.90 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 502.35 | 495.32 | 494.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 510.70 | 501.53 | 497.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 517.00 | 520.98 | 515.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 530.70 | 520.98 | 515.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 546.45 | 526.08 | 518.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:00:00 | 554.85 | 541.88 | 531.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 522.05 | 532.65 | 532.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 522.05 | 532.65 | 532.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 516.50 | 523.78 | 527.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 499.25 | 498.90 | 508.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:45:00 | 500.60 | 498.90 | 508.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 510.90 | 501.30 | 508.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 510.90 | 501.30 | 508.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 508.50 | 502.74 | 508.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 506.25 | 502.74 | 508.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 502.75 | 502.74 | 507.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:45:00 | 499.55 | 502.30 | 507.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 474.57 | 482.41 | 490.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 490.65 | 484.06 | 490.81 | SL hit (close>ema200) qty=0.50 sl=484.06 alert=retest2 |

### Cycle 62 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 439.15 | 432.62 | 431.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 440.65 | 434.23 | 432.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 466.00 | 468.15 | 460.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 466.00 | 468.15 | 460.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 469.40 | 470.26 | 465.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:15:00 | 497.35 | 475.37 | 468.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:30:00 | 490.40 | 494.52 | 493.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 490.60 | 492.43 | 492.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 488.10 | 491.56 | 491.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 488.10 | 491.56 | 491.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 487.75 | 490.80 | 491.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 463.95 | 460.41 | 467.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 465.55 | 462.36 | 465.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 465.55 | 462.36 | 465.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 466.45 | 463.18 | 465.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 458.65 | 463.18 | 465.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 462.40 | 458.98 | 458.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 462.40 | 458.98 | 458.75 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 455.35 | 458.25 | 458.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 453.75 | 456.94 | 457.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 451.80 | 448.11 | 452.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 451.80 | 448.11 | 452.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 449.00 | 448.28 | 451.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 12:45:00 | 446.70 | 447.94 | 451.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 447.50 | 449.59 | 450.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 424.36 | 439.86 | 445.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 425.12 | 439.86 | 445.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 431.50 | 428.21 | 435.66 | SL hit (close>ema200) qty=0.50 sl=428.21 alert=retest2 |

### Cycle 66 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 454.35 | 440.06 | 439.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 471.60 | 452.62 | 446.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 459.65 | 460.22 | 455.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 459.65 | 460.22 | 455.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 506.70 | 513.65 | 506.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 506.70 | 513.65 | 506.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 507.95 | 512.51 | 506.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 511.35 | 511.40 | 507.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 510.25 | 511.28 | 508.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 514.70 | 511.93 | 509.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 520.85 | 519.09 | 518.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 538.50 | 524.14 | 521.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 580.00 | 581.37 | 571.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 575.55 | 581.37 | 571.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 570.25 | 579.15 | 571.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 569.70 | 579.15 | 571.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 574.20 | 578.16 | 571.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 581.40 | 574.19 | 572.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 576.35 | 575.65 | 573.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 581.90 | 573.48 | 573.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 579.50 | 574.84 | 574.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 592.55 | 578.38 | 575.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 575.40 | 578.15 | 578.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 579.45 | 577.59 | 578.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 571.60 | 576.39 | 577.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 567.00 | 572.41 | 575.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-13 09:15:00 | 510.30 | 558.73 | 566.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 536.50 | 525.14 | 523.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 546.45 | 531.65 | 526.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 564.00 | 564.21 | 557.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:00:00 | 564.00 | 564.21 | 557.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 553.30 | 562.03 | 557.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 553.30 | 562.03 | 557.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 556.95 | 561.01 | 557.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 562.55 | 561.01 | 557.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 552.10 | 555.50 | 558.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 547.00 | 545.57 | 549.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:45:00 | 546.70 | 545.57 | 549.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 547.10 | 546.03 | 549.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 547.50 | 546.03 | 549.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 540.90 | 543.80 | 546.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 539.50 | 542.94 | 545.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 550.45 | 543.63 | 544.97 | SL hit (close>static) qty=1.00 sl=548.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 549.75 | 546.03 | 545.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 560.15 | 550.29 | 548.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 605.00 | 607.11 | 597.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 605.00 | 607.11 | 597.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 602.50 | 604.80 | 599.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 596.60 | 604.80 | 599.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 592.55 | 602.35 | 598.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 592.55 | 602.35 | 598.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 594.50 | 600.78 | 598.54 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 590.85 | 596.36 | 596.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 588.30 | 593.57 | 595.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 565.05 | 564.84 | 570.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 565.85 | 564.84 | 570.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 565.80 | 565.55 | 569.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 567.10 | 565.55 | 569.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 591.20 | 569.63 | 570.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 597.40 | 569.63 | 570.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 582.65 | 572.24 | 571.57 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 565.85 | 572.96 | 573.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 561.65 | 570.70 | 571.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 567.50 | 567.26 | 569.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 567.50 | 567.26 | 569.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 571.00 | 568.00 | 569.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 571.85 | 568.00 | 569.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 572.80 | 568.96 | 570.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 571.80 | 568.96 | 570.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 573.05 | 569.78 | 570.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 573.45 | 569.78 | 570.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 579.00 | 571.62 | 571.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 584.90 | 575.83 | 573.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 571.40 | 574.21 | 574.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 565.40 | 572.45 | 573.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 565.05 | 564.02 | 566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 565.30 | 564.14 | 565.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 568.10 | 564.14 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 564.80 | 564.27 | 565.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 564.95 | 564.27 | 565.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 565.30 | 564.10 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 566.00 | 564.10 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 560.05 | 563.29 | 564.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 558.80 | 562.93 | 564.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 559.40 | 562.93 | 564.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 559.30 | 560.19 | 563.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 531.43 | 539.31 | 544.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 530.86 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 531.33 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 537.65 | 537.34 | 542.50 | SL hit (close>ema200) qty=0.50 sl=537.34 alert=retest2 |

### Cycle 78 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 547.90 | 539.21 | 538.86 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 538.60 | 541.17 | 541.42 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 553.30 | 543.29 | 542.33 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 544.00 | 546.63 | 546.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 543.35 | 545.48 | 546.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 526.00 | 519.51 | 524.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 522.70 | 520.13 | 522.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 525.55 | 520.13 | 522.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 529.70 | 522.04 | 523.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 531.35 | 522.04 | 523.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 534.30 | 524.49 | 524.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 536.00 | 526.79 | 525.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 539.25 | 540.62 | 536.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:15:00 | 539.00 | 540.62 | 536.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 536.40 | 539.77 | 536.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 536.40 | 539.77 | 536.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 536.70 | 539.16 | 536.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 541.50 | 537.12 | 536.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 533.30 | 536.25 | 535.81 | SL hit (close<static) qty=1.00 sl=535.10 alert=retest2 |

### Cycle 83 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 529.20 | 534.84 | 535.21 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 536.85 | 535.43 | 535.33 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 532.65 | 534.81 | 535.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 531.75 | 534.19 | 534.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 533.45 | 524.35 | 527.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 533.00 | 526.08 | 527.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 535.35 | 526.08 | 527.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 531.65 | 529.25 | 529.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 535.00 | 530.40 | 529.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 531.55 | 530.34 | 529.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 530.45 | 530.36 | 529.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 529.65 | 530.36 | 529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 529.80 | 530.25 | 529.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 529.55 | 530.25 | 529.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 528.65 | 529.93 | 529.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 528.65 | 529.93 | 529.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 525.90 | 529.12 | 529.27 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 534.45 | 529.89 | 529.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 566.65 | 539.10 | 534.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 585.65 | 587.15 | 578.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 585.65 | 587.15 | 578.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 588.00 | 587.23 | 584.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 600.10 | 587.23 | 584.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 09:15:00 | 660.11 | 627.33 | 608.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 701.65 | 704.21 | 704.47 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 734.25 | 710.05 | 707.07 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 701.40 | 706.43 | 707.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 696.90 | 702.30 | 704.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 698.00 | 697.72 | 701.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 698.00 | 697.72 | 701.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 695.45 | 696.67 | 700.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 691.80 | 696.14 | 699.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 692.55 | 693.15 | 696.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 692.50 | 693.04 | 696.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.21 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.92 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.88 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 673.80 | 667.48 | 676.56 | SL hit (close>ema200) qty=0.50 sl=667.48 alert=retest2 |

### Cycle 92 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 636.90 | 628.05 | 627.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 646.70 | 636.03 | 632.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 642.70 | 644.93 | 641.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 642.70 | 644.93 | 641.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 637.45 | 643.43 | 641.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 637.75 | 643.43 | 641.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 641.70 | 643.08 | 641.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 645.85 | 642.39 | 641.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 623.70 | 636.61 | 640.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 622.30 | 616.83 | 624.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 15:00:00 | 622.30 | 616.83 | 624.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 629.80 | 620.25 | 625.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 636.35 | 620.25 | 625.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 625.95 | 623.44 | 625.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 626.30 | 623.44 | 625.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 626.75 | 624.10 | 625.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 626.75 | 624.10 | 625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 627.25 | 624.73 | 625.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 627.25 | 624.73 | 625.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 626.00 | 624.98 | 625.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 618.70 | 624.98 | 625.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 621.55 | 623.70 | 623.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 617.95 | 622.55 | 623.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 627.55 | 619.93 | 620.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 624.85 | 620.91 | 621.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 626.00 | 620.91 | 621.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 603.00 | 611.67 | 615.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 605.20 | 611.67 | 615.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 610.45 | 594.90 | 597.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 610.60 | 594.90 | 597.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 610.00 | 597.92 | 598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 611.90 | 597.92 | 598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 607.75 | 600.74 | 600.06 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 594.85 | 599.63 | 600.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 589.00 | 597.51 | 599.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 582.50 | 580.80 | 586.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:15:00 | 578.60 | 580.80 | 586.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:30:00 | 578.95 | 580.22 | 585.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 12:30:00 | 578.00 | 579.93 | 584.69 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.67 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 550.00 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.10 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-03 15:15:00 | 556.20 | 554.69 | 561.92 | SL hit (close>ema200) qty=0.50 sl=554.69 alert=retest1 |

### Cycle 98 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 533.20 | 514.45 | 512.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 543.75 | 520.31 | 515.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 528.30 | 535.88 | 536.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 525.20 | 533.75 | 535.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 532.20 | 529.78 | 532.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 532.20 | 529.78 | 532.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 530.55 | 529.94 | 532.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 532.00 | 529.94 | 532.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 531.35 | 530.22 | 532.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 533.40 | 530.22 | 532.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 532.60 | 530.70 | 532.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 532.60 | 530.70 | 532.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 530.60 | 530.68 | 532.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 535.15 | 530.68 | 532.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 540.00 | 532.54 | 532.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 541.80 | 532.54 | 532.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 544.60 | 534.95 | 533.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 551.75 | 541.54 | 537.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 557.30 | 557.97 | 553.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 557.30 | 557.97 | 553.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 561.00 | 563.57 | 559.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 561.00 | 563.57 | 559.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 559.45 | 562.23 | 559.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 559.45 | 562.23 | 559.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 559.35 | 561.65 | 559.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 557.05 | 561.65 | 559.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 553.20 | 559.96 | 559.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 553.20 | 559.96 | 559.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 552.65 | 558.50 | 558.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 545.60 | 555.20 | 557.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 537.25 | 533.59 | 541.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 537.25 | 533.59 | 541.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 545.15 | 535.90 | 541.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 545.15 | 535.90 | 541.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 548.00 | 538.32 | 542.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 541.70 | 539.23 | 542.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 554.90 | 542.36 | 543.24 | SL hit (close>static) qty=1.00 sl=548.40 alert=retest2 |

### Cycle 102 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 556.35 | 545.16 | 544.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 560.65 | 554.01 | 550.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 585.35 | 586.76 | 577.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 585.35 | 586.76 | 577.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 591.90 | 586.58 | 581.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 592.95 | 586.58 | 581.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 592.55 | 587.58 | 582.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 592.05 | 587.58 | 582.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:00:00 | 592.45 | 588.56 | 583.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 597.00 | 591.41 | 586.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 606.60 | 593.26 | 587.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 578.25 | 589.57 | 587.17 | SL hit (close<static) qty=1.00 sl=586.00 alert=retest2 |

### Cycle 103 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 572.80 | 584.58 | 585.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 569.65 | 581.59 | 583.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 558.25 | 552.74 | 561.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 566.15 | 555.42 | 562.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 566.15 | 555.42 | 562.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 571.70 | 558.67 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 566.45 | 558.67 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 565.65 | 561.34 | 563.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 563.80 | 561.34 | 563.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 563.70 | 563.25 | 563.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 560.60 | 562.72 | 563.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.61 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.51 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.57 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 527.05 | 526.98 | 535.93 | SL hit (close>ema200) qty=0.50 sl=526.98 alert=retest2 |

### Cycle 104 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 535.50 | 531.82 | 531.60 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 530.00 | 531.45 | 531.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 524.40 | 530.04 | 530.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 489.10 | 487.26 | 497.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 489.10 | 487.26 | 497.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 492.45 | 489.38 | 496.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 490.20 | 489.38 | 496.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 489.50 | 491.32 | 494.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 514.20 | 505.90 | 501.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 562.85 | 564.10 | 551.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 554.50 | 564.10 | 551.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 564.00 | 564.08 | 552.26 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 545.85 | 551.53 | 551.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 539.20 | 547.91 | 550.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 561.05 | 546.17 | 547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 565.85 | 550.11 | 549.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 574.75 | 563.45 | 557.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 565.50 | 568.25 | 562.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 565.50 | 568.25 | 562.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 565.00 | 567.60 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 564.85 | 567.60 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 555.80 | 565.09 | 562.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 555.80 | 565.09 | 562.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 558.85 | 563.84 | 562.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 561.45 | 563.77 | 562.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 548.65 | 554.29 | 557.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 542.45 | 539.23 | 543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 543.50 | 540.09 | 543.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 544.70 | 540.09 | 543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 543.05 | 540.68 | 543.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 541.00 | 540.68 | 543.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 544.20 | 541.38 | 543.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 543.90 | 541.38 | 543.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 538.35 | 540.78 | 543.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:45:00 | 535.20 | 539.75 | 542.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 560.15 | 547.26 | 545.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 557.20 | 558.35 | 553.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 546.40 | 555.96 | 553.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 546.40 | 555.96 | 553.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 544.20 | 553.61 | 552.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.00 | 553.61 | 552.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 551.70 | 552.09 | 551.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 551.70 | 552.09 | 551.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 549.15 | 551.50 | 551.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 543.75 | 548.15 | 549.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 548.20 | 546.82 | 548.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 548.85 | 547.22 | 548.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 542.45 | 547.22 | 548.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 515.33 | 528.33 | 533.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 13:15:00 | 488.21 | 499.67 | 511.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 112 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 464.85 | 452.00 | 451.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 466.35 | 454.87 | 452.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 446.80 | 454.74 | 455.61 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 462.50 | 456.88 | 456.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 466.50 | 460.54 | 458.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 446.80 | 459.15 | 458.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 445.40 | 456.40 | 457.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 443.40 | 453.80 | 455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 446.30 | 446.14 | 450.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 447.50 | 446.14 | 450.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 454.20 | 447.75 | 450.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 454.20 | 447.75 | 450.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 454.60 | 449.12 | 450.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 454.75 | 449.12 | 450.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 449.20 | 449.10 | 450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 465.00 | 449.10 | 450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 467.85 | 452.85 | 452.05 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 439.95 | 453.21 | 453.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 437.30 | 450.03 | 452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 447.25 | 436.69 | 435.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 454.60 | 442.43 | 439.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 443.60 | 444.01 | 441.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 443.60 | 444.01 | 441.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 454.65 | 451.81 | 446.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 461.40 | 455.06 | 450.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 475.05 | 456.42 | 451.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 507.54 | 491.01 | 485.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 512.10 | 514.31 | 514.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 510.30 | 513.51 | 514.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 493.25 | 489.03 | 498.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 493.25 | 489.03 | 498.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 498.20 | 491.53 | 498.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 498.20 | 491.53 | 498.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 504.30 | 494.09 | 498.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 501.40 | 494.09 | 498.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 501.00 | 495.47 | 498.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 502.40 | 495.47 | 498.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 500.90 | 496.55 | 499.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 498.60 | 496.55 | 499.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 497.50 | 497.17 | 498.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 493.55 | 497.17 | 498.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:15:00 | 494.35 | 496.75 | 498.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 494.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 493.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 493.20 | 494.51 | 496.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 491.50 | 494.55 | 496.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 487.40 | 493.42 | 495.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 507.70 | 491.98 | 492.43 | SL hit (close>static) qty=1.00 sl=499.75 alert=retest2 |

### Cycle 120 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 508.80 | 495.35 | 493.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 515.30 | 506.80 | 501.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 09:15:00 | 377.55 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2024-05-16 12:15:00 | 376.40 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2024-05-16 13:45:00 | 376.30 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-05-16 15:15:00 | 377.00 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-05-17 12:15:00 | 381.20 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-05-17 14:15:00 | 382.00 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-05-27 15:15:00 | 381.40 | 2024-05-31 13:15:00 | 384.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-14 09:15:00 | 423.20 | 2024-06-26 10:15:00 | 439.15 | STOP_HIT | 1.00 | 3.77% |
| SELL | retest2 | 2024-06-27 10:30:00 | 439.10 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-27 12:45:00 | 439.05 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-07-01 10:45:00 | 439.70 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-07-09 15:15:00 | 528.50 | 2024-07-10 09:15:00 | 514.55 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-08-13 12:30:00 | 510.00 | 2024-08-14 14:15:00 | 519.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-08-22 11:30:00 | 595.05 | 2024-08-22 12:15:00 | 586.15 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-23 09:15:00 | 595.60 | 2024-08-23 10:15:00 | 585.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-09-04 14:30:00 | 608.00 | 2024-09-09 10:15:00 | 600.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-19 15:15:00 | 680.00 | 2024-09-20 13:15:00 | 748.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-10-11 13:15:00 | 751.75 | 2024-10-16 09:15:00 | 789.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 13:45:00 | 754.20 | 2024-10-16 09:15:00 | 791.91 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 13:15:00 | 751.75 | 2024-10-16 11:15:00 | 766.25 | STOP_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2024-10-11 13:45:00 | 754.20 | 2024-10-16 11:15:00 | 766.25 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest2 | 2024-10-14 10:30:00 | 756.85 | 2024-10-18 09:15:00 | 752.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-29 10:30:00 | 671.80 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-10-29 11:45:00 | 671.50 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-10-29 12:30:00 | 674.60 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-10-29 13:15:00 | 673.40 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-11-04 13:45:00 | 727.95 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-11-04 14:45:00 | 731.35 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2024-11-05 09:15:00 | 728.10 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-11-05 09:45:00 | 729.40 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-11-11 13:30:00 | 738.30 | 2024-11-13 10:15:00 | 701.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:30:00 | 738.30 | 2024-11-14 13:15:00 | 664.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-29 09:30:00 | 664.00 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-11-29 10:00:00 | 664.40 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-11-29 11:30:00 | 663.65 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-13 13:45:00 | 751.00 | 2024-12-17 09:15:00 | 826.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-13 14:30:00 | 751.50 | 2024-12-17 09:15:00 | 826.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-10 09:15:00 | 910.70 | 2025-01-10 09:15:00 | 882.75 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-01-10 11:00:00 | 912.60 | 2025-01-10 11:15:00 | 896.15 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-23 11:30:00 | 859.10 | 2025-01-24 14:15:00 | 816.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 859.10 | 2025-01-27 09:15:00 | 773.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 09:15:00 | 534.90 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-20 09:45:00 | 532.80 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-02-20 11:45:00 | 537.15 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-20 12:15:00 | 536.70 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-03-07 14:00:00 | 487.00 | 2025-03-17 12:15:00 | 489.85 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-03-11 10:30:00 | 483.75 | 2025-03-17 12:15:00 | 489.85 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-03-24 11:00:00 | 554.85 | 2025-03-25 11:15:00 | 522.05 | STOP_HIT | 1.00 | -5.91% |
| SELL | retest2 | 2025-03-28 10:45:00 | 499.55 | 2025-04-02 09:15:00 | 474.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 10:45:00 | 499.55 | 2025-04-02 10:15:00 | 490.65 | STOP_HIT | 0.50 | 1.78% |
| BUY | retest2 | 2025-04-21 12:15:00 | 497.35 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-04-24 10:30:00 | 490.40 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-04-24 13:15:00 | 490.60 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-04-30 09:15:00 | 458.65 | 2025-05-05 15:15:00 | 462.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-07 12:45:00 | 446.70 | 2025-05-09 09:15:00 | 424.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 447.50 | 2025-05-09 09:15:00 | 425.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 12:45:00 | 446.70 | 2025-05-09 15:15:00 | 431.50 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-05-08 13:00:00 | 447.50 | 2025-05-09 15:15:00 | 431.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-05-12 09:45:00 | 447.50 | 2025-05-12 11:15:00 | 454.35 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-21 10:00:00 | 511.35 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-05-21 13:00:00 | 510.25 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-05-21 14:30:00 | 514.70 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-06-05 09:15:00 | 581.40 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-05 12:30:00 | 576.35 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-06 10:15:00 | 581.90 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-09 09:15:00 | 579.50 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-12 11:45:00 | 567.00 | 2025-06-13 09:15:00 | 510.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 562.55 | 2025-07-02 10:15:00 | 554.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-08 11:00:00 | 539.50 | 2025-07-08 14:15:00 | 550.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-08 15:15:00 | 531.43 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 09:15:00 | 530.86 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 09:15:00 | 531.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2025-09-05 09:45:00 | 541.50 | 2025-09-05 11:15:00 | 533.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-09-19 10:15:00 | 600.10 | 2025-09-22 09:15:00 | 660.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-14 14:15:00 | 657.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-14 14:15:00 | 657.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-14 14:15:00 | 657.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest2 | 2025-11-03 12:15:00 | 645.85 | 2025-11-06 09:15:00 | 627.80 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-11-11 09:15:00 | 618.70 | 2025-11-17 09:15:00 | 634.75 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 11:15:00 | 549.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 11:15:00 | 550.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 11:15:00 | 549.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-12-10 11:30:00 | 504.15 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-12-11 10:45:00 | 505.50 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-12-11 11:45:00 | 505.00 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-12-11 12:45:00 | 505.65 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-31 09:30:00 | 541.70 | 2025-12-31 10:15:00 | 554.90 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-07 10:15:00 | 592.95 | 2026-01-08 12:15:00 | 578.25 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-01-07 10:45:00 | 592.55 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-01-07 11:15:00 | 592.05 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-01-07 12:00:00 | 592.45 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-08 10:30:00 | 606.60 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-20 09:15:00 | 535.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-20 09:15:00 | 535.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-20 09:15:00 | 532.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-01-29 10:15:00 | 490.20 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-01-29 15:00:00 | 489.50 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-02-11 11:45:00 | 561.45 | 2026-02-12 10:15:00 | 551.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-02-17 11:45:00 | 535.20 | 2026-02-18 10:15:00 | 554.50 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-02 09:15:00 | 515.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-04 13:15:00 | 488.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-07 13:45:00 | 461.40 | 2026-04-15 10:15:00 | 507.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 475.05 | 2026-04-21 10:15:00 | 522.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 493.55 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-28 12:15:00 | 494.35 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-28 13:30:00 | 494.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-28 14:00:00 | 493.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-29 14:00:00 | 491.50 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-30 09:15:00 | 487.40 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -4.16% |
