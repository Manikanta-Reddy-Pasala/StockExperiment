# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 24 |
| ALERT3 | 116 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 60 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 48
- **Target hits / Stop hits / Partials:** 0 / 62 / 4
- **Avg / median % per leg:** -0.29% / -0.86%
- **Sum % (uncompounded):** -18.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 5 | 23.8% | 0 | 21 | 0 | -0.72% | -15.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 5 | 23.8% | 0 | 21 | 0 | -0.72% | -15.1% |
| SELL (all) | 45 | 13 | 28.9% | 0 | 41 | 4 | -0.09% | -3.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.23% | -0.5% |
| SELL @ 3rd Alert (retest2) | 43 | 13 | 30.2% | 0 | 39 | 4 | -0.08% | -3.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.23% | -0.5% |
| retest2 (combined) | 64 | 18 | 28.1% | 0 | 60 | 4 | -0.29% | -18.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 549.20 | 554.62 | 555.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 545.20 | 552.74 | 554.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 555.10 | 551.56 | 553.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 554.55 | 551.56 | 553.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 555.75 | 552.40 | 553.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 557.10 | 552.40 | 553.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 551.20 | 551.75 | 552.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 549.50 | 550.92 | 552.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 548.90 | 550.67 | 551.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 547.85 | 550.13 | 551.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 554.55 | 551.02 | 551.77 | SL hit (close>static) qty=1.00 sl=553.55 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 555.55 | 552.63 | 552.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 563.00 | 554.98 | 553.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 10:15:00 | 565.90 | 566.27 | 561.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:00:00 | 565.90 | 566.27 | 561.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 561.60 | 564.47 | 561.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 561.60 | 564.47 | 561.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 563.25 | 564.22 | 562.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 564.45 | 563.98 | 562.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 564.30 | 564.70 | 562.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 562.65 | 565.04 | 565.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 562.65 | 565.04 | 565.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 553.95 | 560.52 | 562.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 558.40 | 556.93 | 559.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 558.40 | 556.93 | 559.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 555.20 | 556.10 | 558.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 553.25 | 556.75 | 557.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 554.10 | 556.04 | 556.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 552.85 | 555.69 | 556.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:45:00 | 553.80 | 554.15 | 555.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 554.50 | 554.22 | 555.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 556.55 | 554.22 | 555.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 556.50 | 554.67 | 555.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 556.50 | 554.67 | 555.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 566.80 | 557.10 | 556.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 566.80 | 557.10 | 556.57 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 551.50 | 556.10 | 556.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 546.85 | 552.00 | 554.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 15:15:00 | 547.80 | 547.64 | 550.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 546.35 | 547.64 | 550.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 521.60 | 520.30 | 523.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 520.25 | 520.73 | 522.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 520.45 | 517.56 | 518.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 518.00 | 515.42 | 515.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 525.00 | 517.43 | 516.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 525.00 | 517.43 | 516.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 536.30 | 523.63 | 520.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 539.30 | 539.78 | 533.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:45:00 | 537.15 | 539.78 | 533.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 545.40 | 546.43 | 541.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 545.40 | 546.43 | 541.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 546.90 | 546.94 | 543.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:30:00 | 542.65 | 546.94 | 543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 587.65 | 584.55 | 580.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 591.45 | 585.80 | 581.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 589.95 | 586.20 | 581.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 566.60 | 580.83 | 580.75 | SL hit (close<static) qty=1.00 sl=570.75 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 568.40 | 578.35 | 579.62 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 582.30 | 578.35 | 578.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 589.50 | 581.34 | 579.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 578.50 | 581.39 | 580.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 578.50 | 581.39 | 580.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 580.20 | 581.15 | 580.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 578.45 | 581.15 | 580.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 579.15 | 580.75 | 579.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 579.15 | 580.75 | 579.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 580.80 | 580.76 | 580.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 581.65 | 580.53 | 580.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 589.90 | 593.45 | 593.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 589.90 | 593.45 | 593.70 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 612.60 | 595.53 | 593.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 590.45 | 597.36 | 598.21 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 602.50 | 597.48 | 597.40 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 593.00 | 596.59 | 597.00 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 599.70 | 597.74 | 597.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 604.60 | 599.59 | 598.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 593.20 | 601.70 | 600.86 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 591.25 | 599.61 | 599.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 587.30 | 595.34 | 597.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 592.30 | 586.53 | 590.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 592.30 | 586.53 | 590.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 594.75 | 588.17 | 590.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 594.75 | 588.17 | 590.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 596.20 | 589.78 | 591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 596.75 | 589.78 | 591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 591.55 | 591.10 | 591.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 591.80 | 591.10 | 591.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 592.15 | 591.31 | 591.49 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 596.10 | 592.27 | 591.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 598.15 | 593.40 | 592.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 590.85 | 593.47 | 592.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 592.90 | 593.47 | 592.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 589.45 | 592.66 | 592.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 588.45 | 592.66 | 592.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 592.25 | 592.58 | 592.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 594.60 | 593.37 | 592.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 590.35 | 592.34 | 592.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 590.35 | 592.34 | 592.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 587.50 | 591.37 | 591.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 591.20 | 584.52 | 587.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 591.20 | 584.52 | 587.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 598.10 | 587.23 | 588.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 555.40 | 587.23 | 588.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 543.45 | 578.48 | 584.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 525.35 | 551.55 | 567.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 525.30 | 535.95 | 548.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 538.50 | 529.70 | 529.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 538.50 | 529.70 | 529.69 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 525.80 | 529.78 | 529.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 524.60 | 528.74 | 529.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 512.00 | 510.75 | 516.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:30:00 | 512.20 | 510.75 | 516.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 512.65 | 510.72 | 513.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 514.05 | 510.72 | 513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 513.50 | 511.28 | 513.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 513.30 | 511.28 | 513.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 512.10 | 511.44 | 513.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 510.20 | 511.57 | 513.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 514.85 | 512.01 | 512.97 | SL hit (close>static) qty=1.00 sl=513.60 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 510.20 | 507.62 | 507.62 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 502.60 | 506.96 | 507.34 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 510.70 | 506.98 | 506.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 514.50 | 509.18 | 507.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 508.10 | 509.54 | 508.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 506.50 | 509.54 | 508.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 508.80 | 509.39 | 508.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 518.25 | 509.25 | 508.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 503.15 | 510.86 | 510.35 | SL hit (close<static) qty=1.00 sl=507.40 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 503.10 | 509.31 | 509.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 501.10 | 505.42 | 507.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 503.80 | 502.93 | 505.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 503.80 | 502.93 | 505.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 506.85 | 503.47 | 504.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 506.85 | 503.47 | 504.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 508.85 | 504.55 | 505.18 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 507.80 | 505.71 | 505.63 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 500.00 | 504.92 | 505.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 11:15:00 | 498.90 | 503.07 | 504.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 501.35 | 501.20 | 502.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:15:00 | 501.70 | 501.20 | 502.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 501.85 | 501.33 | 502.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 500.25 | 501.18 | 502.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 500.20 | 501.18 | 502.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 500.20 | 501.22 | 502.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 506.70 | 502.76 | 502.91 | SL hit (close>static) qty=1.00 sl=505.70 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 507.00 | 503.61 | 503.28 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 502.65 | 504.32 | 504.41 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 506.50 | 504.77 | 504.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 508.20 | 505.55 | 505.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 517.50 | 519.89 | 515.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 517.50 | 519.89 | 515.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 512.95 | 518.50 | 515.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 512.95 | 518.50 | 515.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 514.05 | 517.61 | 515.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 514.05 | 517.61 | 515.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 514.00 | 516.45 | 514.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 514.20 | 516.45 | 514.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 514.10 | 515.98 | 514.87 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 512.70 | 514.18 | 514.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 511.20 | 513.58 | 513.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 458.00 | 457.77 | 468.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 455.00 | 457.77 | 468.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 458.55 | 453.02 | 459.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 455.00 | 453.50 | 458.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 454.20 | 453.72 | 458.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 462.45 | 458.98 | 458.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 462.45 | 458.98 | 458.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 465.90 | 460.36 | 459.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 481.50 | 481.80 | 475.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 481.70 | 481.80 | 475.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 486.50 | 487.21 | 484.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 486.50 | 487.21 | 484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 480.60 | 485.89 | 484.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 480.60 | 485.89 | 484.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 483.30 | 485.37 | 483.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 483.85 | 485.45 | 484.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 483.90 | 484.39 | 484.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 483.95 | 484.01 | 483.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 481.05 | 484.11 | 484.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 481.05 | 484.11 | 484.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 475.50 | 481.36 | 482.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 471.75 | 471.58 | 475.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 472.40 | 471.58 | 475.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 475.65 | 472.39 | 475.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 475.60 | 472.39 | 475.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 477.35 | 473.38 | 475.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 478.20 | 473.38 | 475.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 477.80 | 474.27 | 475.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 477.80 | 474.27 | 475.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 476.55 | 475.50 | 475.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 478.20 | 475.50 | 475.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 478.20 | 476.04 | 476.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 478.35 | 476.04 | 476.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 484.60 | 477.75 | 476.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 489.10 | 480.02 | 477.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 491.30 | 493.46 | 490.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 491.30 | 493.46 | 490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 494.05 | 497.43 | 494.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 494.05 | 497.43 | 494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 494.05 | 496.76 | 494.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 495.15 | 496.76 | 494.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 495.40 | 496.17 | 495.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 515.00 | 516.74 | 516.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 515.00 | 516.74 | 516.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 505.50 | 514.49 | 515.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 513.65 | 511.85 | 513.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 513.65 | 511.85 | 513.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 513.85 | 512.25 | 513.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 507.80 | 512.25 | 513.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 511.30 | 512.06 | 513.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 505.80 | 511.67 | 512.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 503.90 | 506.46 | 509.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 504.60 | 508.30 | 509.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 505.70 | 508.30 | 509.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 497.35 | 492.97 | 494.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 497.35 | 492.97 | 494.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 498.50 | 494.07 | 494.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 498.25 | 494.07 | 494.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-19 15:15:00 | 501.00 | 496.28 | 495.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 501.00 | 496.28 | 495.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 509.80 | 499.90 | 497.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 502.30 | 504.11 | 501.36 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 496.75 | 499.92 | 500.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 494.50 | 498.69 | 499.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 492.75 | 486.05 | 489.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 492.75 | 486.05 | 489.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 494.30 | 487.70 | 489.84 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 496.60 | 491.54 | 491.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 497.90 | 492.81 | 491.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 492.80 | 493.96 | 492.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 492.80 | 493.96 | 492.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 490.60 | 493.29 | 492.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 490.60 | 493.29 | 492.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 492.00 | 493.03 | 492.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 493.00 | 493.13 | 492.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 493.50 | 493.13 | 492.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 495.55 | 501.27 | 501.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 495.55 | 501.27 | 501.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 494.00 | 499.81 | 501.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 491.20 | 490.93 | 493.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:45:00 | 490.00 | 490.93 | 493.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 492.00 | 490.91 | 492.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 491.00 | 490.91 | 492.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 492.05 | 491.14 | 492.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 492.95 | 491.14 | 492.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 487.70 | 490.45 | 492.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 485.75 | 489.26 | 491.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 461.46 | 477.32 | 483.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 469.20 | 467.21 | 471.83 | SL hit (close>ema200) qty=0.50 sl=467.21 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 477.20 | 472.95 | 472.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 479.20 | 474.20 | 473.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 478.60 | 479.79 | 477.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 475.65 | 479.79 | 477.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 473.55 | 478.54 | 477.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 475.30 | 478.54 | 477.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 474.35 | 477.70 | 476.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 477.45 | 477.82 | 477.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 473.35 | 476.65 | 476.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 473.35 | 476.65 | 476.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 472.25 | 474.98 | 476.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 473.50 | 471.86 | 473.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 473.70 | 471.86 | 473.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 474.00 | 472.29 | 473.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 476.05 | 472.29 | 473.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 476.15 | 473.06 | 473.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 476.15 | 473.06 | 473.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 474.80 | 473.41 | 473.78 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 476.50 | 474.03 | 474.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 485.20 | 476.26 | 475.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 488.85 | 488.98 | 486.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 488.85 | 488.98 | 486.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 486.85 | 488.49 | 486.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 486.85 | 488.49 | 486.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 485.85 | 487.96 | 486.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 485.85 | 487.96 | 486.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 487.75 | 487.92 | 486.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 492.50 | 487.92 | 486.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 492.35 | 488.80 | 486.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 496.40 | 491.55 | 489.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 486.80 | 488.85 | 488.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 486.80 | 488.85 | 488.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 483.60 | 487.39 | 488.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 484.95 | 484.59 | 486.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 484.95 | 484.59 | 486.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 483.50 | 483.87 | 485.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:00:00 | 482.15 | 484.33 | 485.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 481.85 | 483.72 | 484.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 490.75 | 485.24 | 485.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 490.75 | 485.24 | 485.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 494.75 | 487.14 | 485.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 499.70 | 501.46 | 498.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 507.35 | 501.06 | 498.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 504.75 | 509.43 | 509.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 504.75 | 509.43 | 509.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 497.40 | 506.09 | 508.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 501.10 | 500.78 | 504.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 502.80 | 500.78 | 504.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 495.20 | 499.78 | 503.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 491.25 | 496.77 | 500.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 492.00 | 495.81 | 500.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 485.35 | 495.19 | 498.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:15:00 | 467.40 | 479.72 | 487.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:15:00 | 466.69 | 477.09 | 485.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 461.08 | 466.97 | 476.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 461.45 | 458.87 | 468.66 | SL hit (close>ema200) qty=0.50 sl=458.87 alert=retest2 |

### Cycle 44 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 374.70 | 366.55 | 366.44 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 362.15 | 365.89 | 366.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 357.50 | 364.21 | 365.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 365.55 | 362.26 | 363.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 365.55 | 362.26 | 363.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 367.95 | 363.40 | 364.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 383.95 | 363.40 | 364.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 383.80 | 367.48 | 366.01 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 366.30 | 376.33 | 377.55 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 421.55 | 385.35 | 380.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 425.70 | 393.42 | 384.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 429.25 | 430.77 | 419.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 429.25 | 430.77 | 419.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 426.90 | 427.96 | 423.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:00:00 | 432.20 | 427.59 | 424.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 432.75 | 428.62 | 425.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 420.45 | 427.53 | 425.28 | SL hit (close<static) qty=1.00 sl=423.15 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 419.45 | 423.18 | 423.63 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 427.50 | 424.20 | 423.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 428.30 | 425.02 | 424.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.15 | 426.30 | 425.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 423.45 | 426.30 | 425.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 422.30 | 425.50 | 424.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 422.30 | 425.50 | 424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 422.10 | 424.82 | 424.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:00:00 | 425.75 | 425.01 | 424.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 421.90 | 424.38 | 424.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 421.90 | 424.38 | 424.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 420.75 | 423.66 | 424.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 418.15 | 417.76 | 420.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:45:00 | 417.60 | 417.76 | 420.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 412.15 | 416.65 | 419.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 411.80 | 416.65 | 419.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 407.10 | 405.23 | 405.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 407.10 | 405.23 | 405.11 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 403.60 | 404.95 | 405.01 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 408.25 | 405.64 | 405.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 413.80 | 408.06 | 406.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 408.65 | 411.05 | 408.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 408.65 | 411.05 | 408.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 409.75 | 410.79 | 408.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 402.45 | 410.79 | 408.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 405.05 | 409.64 | 408.55 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 395.90 | 405.60 | 406.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 393.70 | 403.22 | 405.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 402.40 | 401.79 | 404.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 402.40 | 401.79 | 404.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 392.25 | 393.52 | 397.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 391.20 | 393.52 | 397.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 399.55 | 394.20 | 396.25 | SL hit (close>static) qty=1.00 sl=399.45 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 397.55 | 392.40 | 392.06 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 389.75 | 392.09 | 392.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 388.60 | 391.39 | 391.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 390.80 | 389.72 | 390.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 390.80 | 389.72 | 390.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 391.35 | 390.04 | 390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 392.30 | 390.04 | 390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 391.95 | 390.43 | 390.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 392.60 | 390.43 | 390.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 389.70 | 390.28 | 390.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 382.30 | 390.12 | 390.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 385.00 | 381.88 | 381.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 385.00 | 381.88 | 381.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 388.70 | 384.24 | 383.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 376.45 | 384.82 | 383.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 376.45 | 384.82 | 383.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 376.80 | 383.21 | 383.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 375.55 | 380.09 | 381.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 387.30 | 379.92 | 381.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 385.55 | 379.92 | 381.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 387.10 | 381.35 | 381.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 387.65 | 381.35 | 381.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 384.60 | 382.00 | 381.89 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 368.20 | 379.84 | 381.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 363.45 | 376.56 | 379.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 368.60 | 367.38 | 371.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 368.60 | 367.38 | 371.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 373.40 | 368.58 | 371.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 372.85 | 368.58 | 371.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 374.55 | 369.78 | 372.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 374.55 | 369.78 | 372.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 372.15 | 370.49 | 372.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 383.45 | 370.49 | 372.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 388.80 | 374.15 | 373.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 396.65 | 381.43 | 377.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 385.10 | 387.63 | 382.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 385.30 | 387.63 | 382.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 379.95 | 387.32 | 385.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 379.95 | 387.32 | 385.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 378.05 | 385.47 | 384.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 378.05 | 385.47 | 384.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 379.35 | 383.39 | 383.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 376.30 | 381.22 | 382.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 387.80 | 381.75 | 382.56 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 390.70 | 384.31 | 383.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 395.50 | 387.61 | 385.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 414.00 | 414.27 | 406.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 414.00 | 414.27 | 406.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 440.10 | 446.28 | 441.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 446.45 | 442.88 | 441.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 439.75 | 441.68 | 441.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 439.75 | 441.68 | 441.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 10:15:00 | 418.75 | 436.93 | 439.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 414.85 | 414.66 | 418.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:15:00 | 410.60 | 414.66 | 418.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 14:00:00 | 412.40 | 411.91 | 415.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 410.25 | 411.79 | 414.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 407.70 | 411.79 | 414.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 412.45 | 408.13 | 410.58 | SL hit (close>ema400) qty=1.00 sl=410.58 alert=retest1 |

### Cycle 66 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 413.60 | 411.94 | 411.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 418.90 | 414.02 | 413.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 412.90 | 414.90 | 413.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:15:00 | 409.95 | 414.90 | 413.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 412.30 | 414.38 | 413.82 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 405.75 | 412.35 | 412.98 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 416.45 | 412.47 | 412.37 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 406.10 | 411.76 | 412.34 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 413.85 | 411.44 | 411.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 415.45 | 412.24 | 411.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 407.65 | 411.96 | 411.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 407.65 | 411.96 | 411.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 408.15 | 411.20 | 411.33 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 428.00 | 413.95 | 412.23 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 10:45:00 | 549.50 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-22 12:45:00 | 548.90 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-22 13:30:00 | 547.85 | 2025-05-22 14:15:00 | 554.55 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-23 10:15:00 | 549.40 | 2025-05-23 12:15:00 | 553.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-28 09:15:00 | 564.45 | 2025-05-30 10:15:00 | 562.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-28 12:00:00 | 564.30 | 2025-05-30 10:15:00 | 562.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-06-05 09:15:00 | 553.25 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-06-05 12:00:00 | 554.10 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-06-05 15:15:00 | 552.85 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-06 10:45:00 | 553.80 | 2025-06-06 13:15:00 | 566.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-06-17 11:30:00 | 520.25 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-19 10:00:00 | 520.45 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-20 15:00:00 | 518.00 | 2025-06-24 09:15:00 | 525.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-07 10:30:00 | 591.45 | 2025-07-08 09:15:00 | 566.60 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2025-07-07 11:30:00 | 589.95 | 2025-07-08 09:15:00 | 566.60 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-07-11 14:15:00 | 581.65 | 2025-07-18 15:15:00 | 589.90 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-08-06 12:30:00 | 594.60 | 2025-08-06 14:15:00 | 590.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-08 15:00:00 | 525.35 | 2025-08-14 13:15:00 | 538.50 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-08-12 09:15:00 | 525.30 | 2025-08-14 13:15:00 | 538.50 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-21 15:15:00 | 510.20 | 2025-08-22 09:15:00 | 514.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-22 13:45:00 | 510.40 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-08-25 09:15:00 | 510.55 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-25 14:15:00 | 510.40 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-08-26 11:30:00 | 501.65 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-08-26 13:15:00 | 501.50 | 2025-08-28 13:15:00 | 513.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 518.25 | 2025-09-04 09:15:00 | 503.15 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-10 10:30:00 | 500.25 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-10 11:15:00 | 500.20 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-10 12:30:00 | 500.20 | 2025-09-10 14:15:00 | 506.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-30 10:30:00 | 455.00 | 2025-10-01 13:15:00 | 462.45 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-09-30 11:30:00 | 454.20 | 2025-10-01 13:15:00 | 462.45 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-10-09 09:30:00 | 483.85 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-09 13:45:00 | 483.90 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-09 15:15:00 | 483.95 | 2025-10-13 09:15:00 | 481.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-24 09:15:00 | 495.15 | 2025-11-06 15:15:00 | 515.00 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2025-10-24 15:00:00 | 495.40 | 2025-11-06 15:15:00 | 515.00 | STOP_HIT | 1.00 | 3.96% |
| SELL | retest2 | 2025-11-11 09:15:00 | 505.80 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-11-11 12:45:00 | 503.90 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-11-13 09:30:00 | 504.60 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-11-13 10:00:00 | 505.70 | 2025-11-19 15:15:00 | 501.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-11-27 14:30:00 | 493.00 | 2025-12-03 09:15:00 | 495.55 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-11-27 15:00:00 | 493.50 | 2025-12-03 09:15:00 | 495.55 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-12-08 10:30:00 | 485.75 | 2025-12-09 09:15:00 | 461.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:30:00 | 485.75 | 2025-12-11 10:15:00 | 469.20 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-12-16 11:30:00 | 477.45 | 2025-12-17 12:15:00 | 473.35 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-29 09:30:00 | 496.40 | 2025-12-29 14:15:00 | 486.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-01 11:00:00 | 482.15 | 2026-01-02 09:15:00 | 490.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-01 13:00:00 | 481.85 | 2026-01-02 09:15:00 | 490.75 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-07 09:15:00 | 507.35 | 2026-01-09 13:15:00 | 504.75 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-13 12:45:00 | 491.25 | 2026-01-16 10:15:00 | 467.40 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-13 14:00:00 | 492.00 | 2026-01-16 11:15:00 | 466.69 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-19 10:15:00 | 461.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 491.25 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2026-01-13 14:00:00 | 492.00 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-19 14:15:00 | 461.45 | STOP_HIT | 0.50 | 4.92% |
| BUY | retest2 | 2026-02-12 14:00:00 | 432.20 | 2026-02-13 09:15:00 | 420.45 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-02-12 15:00:00 | 432.75 | 2026-02-13 09:15:00 | 420.45 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-17 13:00:00 | 425.75 | 2026-02-17 13:15:00 | 421.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-19 10:15:00 | 411.80 | 2026-02-25 10:15:00 | 407.10 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2026-03-05 10:15:00 | 391.20 | 2026-03-05 14:15:00 | 399.55 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-03-09 09:15:00 | 384.10 | 2026-03-11 09:15:00 | 397.55 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-03-13 09:15:00 | 382.30 | 2026-03-18 09:15:00 | 385.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-04-15 09:30:00 | 446.45 | 2026-04-16 11:15:00 | 439.75 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest1 | 2026-04-23 09:15:00 | 410.60 | 2026-04-27 09:15:00 | 412.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-23 14:00:00 | 412.40 | 2026-04-27 09:15:00 | 412.45 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-04-24 10:15:00 | 407.70 | 2026-04-27 13:15:00 | 413.60 | STOP_HIT | 1.00 | -1.45% |
