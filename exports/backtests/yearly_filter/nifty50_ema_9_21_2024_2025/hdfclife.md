# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 619.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 104 |
| ALERT2 | 104 |
| ALERT2_SKIP | 51 |
| ALERT3 | 255 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 121 |
| PARTIAL | 12 |
| TARGET_HIT | 0 |
| STOP_HIT | 125 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 89
- **Target hits / Stop hits / Partials:** 0 / 125 / 12
- **Avg / median % per leg:** 0.36% / -0.33%
- **Sum % (uncompounded):** 49.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 18 | 29.0% | 0 | 60 | 2 | 0.14% | 8.8% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.70% | 18.8% |
| BUY @ 3rd Alert (retest2) | 58 | 14 | 24.1% | 0 | 58 | 0 | -0.17% | -10.0% |
| SELL (all) | 75 | 30 | 40.0% | 0 | 65 | 10 | 0.54% | 40.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.08% | -2.2% |
| SELL @ 3rd Alert (retest2) | 73 | 30 | 41.1% | 0 | 63 | 10 | 0.58% | 42.6% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.77% | 16.6% |
| retest2 (combined) | 131 | 44 | 33.6% | 0 | 121 | 10 | 0.25% | 32.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 09:15:00 | 559.05 | 550.18 | 549.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 568.20 | 559.04 | 555.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 14:15:00 | 562.05 | 562.59 | 558.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 14:30:00 | 564.60 | 562.59 | 558.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 560.00 | 561.79 | 559.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:30:00 | 558.50 | 561.79 | 559.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 558.70 | 561.17 | 559.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 558.80 | 561.17 | 559.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 556.80 | 560.30 | 558.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 556.70 | 560.30 | 558.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 556.45 | 559.53 | 558.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 556.40 | 559.53 | 558.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 555.95 | 558.20 | 558.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 553.00 | 556.80 | 557.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 558.75 | 556.58 | 557.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 559.95 | 557.26 | 557.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 559.95 | 557.26 | 557.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 558.70 | 557.54 | 557.63 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 566.65 | 559.37 | 558.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 569.10 | 562.39 | 560.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 565.35 | 568.41 | 565.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 565.70 | 567.86 | 565.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 565.70 | 567.86 | 565.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 564.45 | 567.18 | 565.52 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 563.20 | 564.76 | 564.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 561.45 | 563.86 | 564.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 564.35 | 562.48 | 563.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 563.70 | 562.72 | 563.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:45:00 | 564.85 | 562.72 | 563.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 568.50 | 563.88 | 563.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 568.50 | 563.88 | 563.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 565.95 | 564.29 | 564.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 571.00 | 566.48 | 565.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 566.15 | 566.88 | 565.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 565.10 | 566.53 | 565.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 565.10 | 566.53 | 565.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 565.20 | 566.26 | 565.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 566.70 | 566.26 | 565.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 566.75 | 566.12 | 565.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 14:15:00 | 564.05 | 566.32 | 566.00 | SL hit (close<static) qty=1.00 sl=564.40 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 563.00 | 568.64 | 569.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 561.80 | 567.27 | 568.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 15:15:00 | 551.00 | 550.47 | 553.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 539.45 | 550.47 | 553.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 550.50 | 550.48 | 552.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 533.95 | 547.17 | 551.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:30:00 | 536.20 | 542.62 | 548.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:15:00 | 535.10 | 539.36 | 546.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:45:00 | 534.85 | 538.40 | 544.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 544.90 | 539.24 | 544.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 544.90 | 539.24 | 544.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 548.75 | 541.14 | 544.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 548.75 | 541.14 | 544.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 550.35 | 542.98 | 545.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 550.35 | 542.98 | 545.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 551.90 | 546.90 | 546.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 551.90 | 546.90 | 546.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 555.15 | 549.19 | 547.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 571.35 | 571.42 | 566.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 571.35 | 571.42 | 566.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 588.75 | 576.47 | 572.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 592.30 | 579.43 | 574.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 592.00 | 584.93 | 578.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 596.30 | 587.58 | 580.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 10:15:00 | 590.05 | 595.22 | 595.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 590.05 | 595.22 | 595.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 585.00 | 590.31 | 592.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 582.45 | 581.08 | 584.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 10:00:00 | 582.45 | 581.08 | 584.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 584.30 | 581.16 | 583.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 584.30 | 581.16 | 583.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 585.00 | 581.93 | 583.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 585.00 | 581.93 | 583.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 591.55 | 583.85 | 584.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 591.55 | 583.85 | 584.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 589.45 | 584.97 | 584.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 593.20 | 586.62 | 585.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 588.50 | 589.05 | 587.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 13:00:00 | 588.50 | 589.05 | 587.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 588.55 | 588.95 | 587.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 587.30 | 588.95 | 587.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 594.30 | 590.09 | 588.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 595.35 | 591.10 | 589.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 10:00:00 | 596.45 | 592.17 | 590.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:45:00 | 595.70 | 593.74 | 591.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:30:00 | 596.55 | 594.33 | 592.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 587.40 | 595.94 | 594.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 587.40 | 595.94 | 594.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 594.60 | 595.67 | 594.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 11:45:00 | 595.35 | 594.87 | 594.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 589.55 | 593.80 | 593.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 589.55 | 593.80 | 593.88 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 597.30 | 594.13 | 593.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 603.70 | 596.76 | 595.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 621.55 | 621.97 | 616.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 622.65 | 621.97 | 616.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 622.05 | 621.93 | 617.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 619.15 | 621.93 | 617.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 629.50 | 632.38 | 628.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 628.50 | 632.38 | 628.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 637.55 | 635.22 | 632.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 639.00 | 636.18 | 633.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 647.10 | 638.16 | 635.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 10:30:00 | 639.30 | 638.46 | 635.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 637.25 | 642.02 | 642.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 637.25 | 642.02 | 642.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 635.60 | 640.74 | 641.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 640.00 | 639.63 | 640.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 640.00 | 639.63 | 640.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 637.90 | 639.29 | 640.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 636.25 | 638.88 | 640.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 636.25 | 638.35 | 639.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 632.85 | 636.79 | 638.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 624.20 | 637.63 | 638.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 637.10 | 637.52 | 638.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 641.20 | 637.52 | 638.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 642.90 | 638.60 | 639.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 642.90 | 638.60 | 639.04 | SL hit (close>static) qty=1.00 sl=640.85 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 644.50 | 639.78 | 639.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 663.70 | 644.92 | 641.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 696.20 | 697.71 | 685.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:45:00 | 697.65 | 697.71 | 685.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 686.15 | 695.96 | 690.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 686.15 | 695.96 | 690.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 696.35 | 696.03 | 691.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 699.60 | 696.03 | 691.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 698.95 | 698.36 | 694.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 698.40 | 706.80 | 707.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 698.40 | 706.80 | 707.59 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 15:15:00 | 708.90 | 707.24 | 707.21 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 10:15:00 | 696.90 | 705.23 | 706.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 688.00 | 699.39 | 703.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 692.70 | 692.32 | 697.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:30:00 | 692.45 | 692.32 | 697.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 696.95 | 692.31 | 696.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 696.95 | 692.31 | 696.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 695.85 | 693.02 | 695.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 692.00 | 693.02 | 695.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 691.90 | 692.67 | 695.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 703.60 | 694.86 | 696.06 | SL hit (close>static) qty=1.00 sl=697.15 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 709.90 | 697.87 | 697.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 13:15:00 | 711.20 | 700.53 | 698.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 704.15 | 705.27 | 702.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 704.15 | 705.27 | 702.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 701.90 | 704.91 | 702.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 701.90 | 704.91 | 702.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 702.60 | 704.45 | 702.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 699.20 | 704.45 | 702.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 695.45 | 702.65 | 702.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 695.45 | 702.65 | 702.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 703.80 | 703.30 | 702.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:45:00 | 702.60 | 703.30 | 702.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 702.15 | 703.41 | 702.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 702.15 | 703.41 | 702.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 701.00 | 702.93 | 702.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 697.10 | 702.93 | 702.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 693.05 | 700.96 | 701.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 10:15:00 | 691.30 | 699.02 | 700.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 688.55 | 686.90 | 691.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 688.55 | 686.90 | 691.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 683.35 | 682.99 | 687.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 680.70 | 686.22 | 687.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 11:00:00 | 681.05 | 684.59 | 686.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 696.95 | 687.44 | 687.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 696.95 | 687.44 | 687.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 701.40 | 690.23 | 688.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 726.95 | 727.33 | 723.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 11:00:00 | 726.95 | 727.33 | 723.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 723.90 | 726.55 | 723.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 723.90 | 726.55 | 723.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 725.90 | 726.42 | 724.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 728.80 | 726.59 | 724.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 729.15 | 727.63 | 725.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 743.55 | 750.72 | 751.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 743.55 | 750.72 | 751.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 737.35 | 744.79 | 747.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 747.00 | 744.19 | 746.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 745.35 | 744.42 | 746.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 745.75 | 744.42 | 746.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 710.65 | 704.63 | 710.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 710.65 | 704.63 | 710.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 715.35 | 706.77 | 710.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 715.35 | 706.77 | 710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 711.65 | 707.75 | 710.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 708.70 | 707.55 | 710.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 706.05 | 701.56 | 701.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 706.05 | 701.56 | 701.33 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 699.25 | 701.18 | 701.20 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 718.75 | 704.52 | 702.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 721.90 | 713.11 | 708.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 719.10 | 721.03 | 715.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 719.10 | 721.03 | 715.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 722.55 | 720.88 | 717.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:15:00 | 726.95 | 718.00 | 717.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:00:00 | 727.05 | 719.81 | 718.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 726.65 | 721.80 | 719.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 727.70 | 722.51 | 719.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 730.75 | 731.90 | 727.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 730.75 | 731.90 | 727.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 725.85 | 730.87 | 727.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 725.85 | 730.87 | 727.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 722.15 | 729.13 | 727.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 722.15 | 729.13 | 727.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 723.35 | 727.97 | 727.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 717.80 | 725.94 | 726.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 717.80 | 725.94 | 726.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 714.00 | 720.55 | 723.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 708.05 | 706.48 | 711.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:45:00 | 709.40 | 706.48 | 711.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 712.95 | 707.46 | 710.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 712.95 | 707.46 | 710.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 709.30 | 707.83 | 710.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 705.60 | 707.16 | 710.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 716.00 | 709.47 | 710.48 | SL hit (close>static) qty=1.00 sl=713.50 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 718.70 | 709.12 | 708.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 722.80 | 711.86 | 709.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 717.00 | 717.05 | 713.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:45:00 | 718.25 | 717.05 | 713.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 715.40 | 716.38 | 714.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 713.95 | 716.38 | 714.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 720.60 | 720.30 | 717.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 718.65 | 720.30 | 717.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 715.15 | 734.51 | 731.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 715.15 | 734.51 | 731.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 714.05 | 730.42 | 730.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 727.00 | 730.42 | 730.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 723.40 | 729.02 | 729.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 723.40 | 729.02 | 729.42 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 737.00 | 728.89 | 727.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 739.20 | 730.96 | 728.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 15:15:00 | 742.45 | 744.26 | 739.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 09:15:00 | 746.85 | 744.26 | 739.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 747.60 | 744.93 | 740.46 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 726.95 | 738.82 | 739.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 723.60 | 734.32 | 736.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 710.00 | 709.16 | 715.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:15:00 | 710.30 | 709.16 | 715.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 713.50 | 710.03 | 715.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 715.20 | 710.03 | 715.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 720.60 | 712.14 | 715.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 720.60 | 712.14 | 715.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 716.25 | 712.96 | 715.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 718.60 | 712.96 | 715.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 716.45 | 713.66 | 716.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 716.70 | 713.66 | 716.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 715.65 | 714.06 | 715.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:45:00 | 717.60 | 714.06 | 715.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 718.15 | 714.88 | 716.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 720.10 | 714.88 | 716.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 719.70 | 715.84 | 716.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 716.25 | 715.84 | 716.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 10:15:00 | 723.90 | 717.65 | 717.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 723.90 | 717.65 | 717.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 728.45 | 719.81 | 718.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 731.55 | 733.71 | 728.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 731.55 | 733.71 | 728.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 729.00 | 732.77 | 728.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 729.00 | 732.77 | 728.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 726.50 | 731.51 | 728.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 726.50 | 731.51 | 728.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 725.00 | 730.21 | 728.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 718.10 | 730.21 | 728.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 721.15 | 726.56 | 726.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 711.45 | 719.53 | 722.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 724.15 | 707.40 | 712.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 716.75 | 709.27 | 712.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 715.50 | 709.27 | 712.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 711.70 | 710.53 | 712.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 717.10 | 712.03 | 711.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 717.10 | 712.03 | 711.46 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 707.40 | 710.81 | 711.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 703.25 | 709.30 | 710.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 710.00 | 709.44 | 710.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 707.90 | 709.13 | 710.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 710.85 | 709.13 | 710.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 710.10 | 709.33 | 710.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:45:00 | 707.35 | 708.19 | 709.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:15:00 | 707.75 | 707.93 | 709.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 706.90 | 708.00 | 708.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 671.98 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 672.36 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 671.55 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 675.35 | 675.30 | 679.92 | SL hit (close>ema200) qty=0.50 sl=675.30 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 689.45 | 682.51 | 681.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 690.05 | 684.02 | 682.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 683.25 | 686.66 | 684.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 688.35 | 686.99 | 685.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:30:00 | 689.60 | 686.80 | 685.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:30:00 | 690.95 | 687.55 | 685.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:00:00 | 689.55 | 687.95 | 686.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 14:15:00 | 682.90 | 686.66 | 686.10 | SL hit (close<static) qty=1.00 sl=683.10 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 681.50 | 685.63 | 685.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 681.00 | 684.00 | 684.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 681.25 | 683.84 | 684.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:45:00 | 669.90 | 678.56 | 682.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 13:15:00 | 647.19 | 671.58 | 678.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 658.00 | 657.85 | 666.09 | SL hit (close>ema200) qty=0.50 sl=657.85 alert=retest2 |

### Cycle 35 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 634.60 | 632.02 | 631.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 628.65 | 631.89 | 632.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 627.80 | 631.07 | 631.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 623.40 | 622.38 | 624.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 623.40 | 622.38 | 624.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 624.10 | 622.73 | 624.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 619.65 | 622.73 | 624.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 623.00 | 622.57 | 624.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:00:00 | 623.00 | 622.40 | 623.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:45:00 | 622.65 | 622.39 | 623.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 623.75 | 622.66 | 623.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 617.20 | 622.66 | 623.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 620.85 | 622.24 | 623.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 620.50 | 622.01 | 622.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 626.15 | 623.18 | 623.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 626.15 | 623.18 | 623.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 628.55 | 624.65 | 623.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 624.35 | 624.81 | 624.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 626.10 | 625.07 | 624.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 625.40 | 625.07 | 624.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 624.30 | 625.23 | 624.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 624.30 | 625.23 | 624.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 624.35 | 625.05 | 624.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 623.20 | 625.05 | 624.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 624.70 | 624.98 | 624.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 623.55 | 624.98 | 624.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 622.95 | 624.58 | 624.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 624.10 | 624.58 | 624.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 621.65 | 623.99 | 624.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 620.80 | 623.35 | 623.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 616.85 | 616.74 | 619.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:00:00 | 616.85 | 616.74 | 619.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 618.75 | 617.18 | 618.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 618.75 | 617.18 | 618.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 618.25 | 617.39 | 618.46 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 619.95 | 618.36 | 618.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 620.35 | 618.76 | 618.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 616.80 | 618.56 | 618.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 617.35 | 618.32 | 618.35 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 622.20 | 619.10 | 618.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 14:15:00 | 623.35 | 620.79 | 619.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 614.60 | 620.20 | 619.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 609.50 | 618.06 | 618.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 605.90 | 613.33 | 616.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 612.30 | 610.40 | 613.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 619.45 | 612.21 | 613.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 619.45 | 612.21 | 613.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 618.40 | 613.45 | 614.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:15:00 | 619.20 | 613.45 | 614.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 619.95 | 615.76 | 615.24 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 613.15 | 616.07 | 616.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 604.95 | 612.59 | 614.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:15:00 | 612.65 | 612.50 | 613.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 611.65 | 612.33 | 613.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:15:00 | 610.40 | 612.33 | 613.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 653.10 | 605.86 | 602.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 653.10 | 605.86 | 602.89 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 627.65 | 630.98 | 631.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 624.55 | 629.09 | 630.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 619.45 | 619.12 | 622.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 12:15:00 | 624.60 | 620.53 | 622.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 624.60 | 620.53 | 622.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 623.75 | 620.53 | 622.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 623.15 | 621.05 | 622.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:45:00 | 621.35 | 621.04 | 622.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 620.85 | 620.57 | 621.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 621.05 | 620.67 | 621.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 622.55 | 614.09 | 613.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 622.55 | 614.09 | 613.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 627.85 | 617.88 | 615.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 634.60 | 634.75 | 629.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 634.60 | 634.75 | 629.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 599.85 | 631.39 | 630.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 599.85 | 631.39 | 630.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 620.50 | 629.21 | 629.87 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 635.60 | 626.20 | 625.31 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 624.50 | 632.28 | 632.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 622.95 | 629.24 | 630.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 627.30 | 621.25 | 625.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 633.05 | 623.61 | 625.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 632.25 | 623.61 | 625.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 629.15 | 627.57 | 627.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 636.20 | 629.61 | 628.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:00:00 | 629.30 | 632.51 | 630.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 629.50 | 631.90 | 630.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 632.75 | 631.38 | 630.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 09:15:00 | 627.75 | 630.66 | 630.08 | SL hit (close<static) qty=1.00 sl=628.45 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 624.55 | 628.73 | 629.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 623.70 | 627.72 | 628.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 622.70 | 622.47 | 624.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:15:00 | 624.00 | 622.47 | 624.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 624.00 | 622.77 | 624.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 615.90 | 622.77 | 624.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 615.25 | 621.27 | 623.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 612.60 | 619.54 | 622.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 623.50 | 621.39 | 621.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 623.50 | 621.39 | 621.24 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 619.00 | 620.89 | 621.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 617.75 | 620.14 | 620.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 620.55 | 620.01 | 620.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 620.40 | 620.08 | 620.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 620.90 | 620.08 | 620.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 614.40 | 618.95 | 619.94 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 622.95 | 620.20 | 620.14 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 617.70 | 619.96 | 620.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 615.30 | 617.43 | 618.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 620.85 | 617.25 | 618.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 622.50 | 618.30 | 618.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 620.00 | 618.30 | 618.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 15:15:00 | 620.00 | 618.64 | 618.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 15:15:00 | 620.00 | 618.64 | 618.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 10:15:00 | 622.75 | 619.94 | 619.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:30:00 | 619.95 | 620.10 | 619.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 618.25 | 619.73 | 619.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 618.15 | 619.73 | 619.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 618.90 | 619.56 | 619.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:15:00 | 615.00 | 619.56 | 619.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 15:15:00 | 615.00 | 618.65 | 618.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 603.95 | 615.71 | 617.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 618.10 | 611.29 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 611.80 | 611.39 | 613.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 615.10 | 611.39 | 613.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 615.85 | 612.28 | 613.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:00:00 | 615.85 | 612.28 | 613.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 616.10 | 613.05 | 614.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 616.05 | 613.05 | 614.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 618.10 | 614.68 | 614.64 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 613.25 | 614.61 | 614.69 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 616.20 | 614.93 | 614.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 620.30 | 616.47 | 615.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 618.70 | 619.77 | 618.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 615.10 | 618.83 | 617.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 615.10 | 618.83 | 617.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:45:00 | 615.80 | 618.83 | 617.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 616.60 | 618.39 | 617.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 13:00:00 | 617.85 | 617.91 | 617.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:15:00 | 617.60 | 617.71 | 617.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 631.55 | 617.75 | 617.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 12:15:00 | 622.80 | 628.83 | 629.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 622.80 | 628.83 | 629.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 621.85 | 626.40 | 628.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 631.00 | 625.68 | 627.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 626.15 | 625.78 | 627.37 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 629.30 | 628.18 | 628.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 632.75 | 629.09 | 628.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 675.35 | 679.20 | 675.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 676.25 | 678.61 | 675.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 675.85 | 678.61 | 675.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 678.65 | 678.62 | 675.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 676.65 | 678.62 | 675.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 672.35 | 677.36 | 675.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 672.35 | 677.36 | 675.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 672.90 | 676.47 | 675.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 677.55 | 676.47 | 675.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 673.20 | 675.89 | 675.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 673.20 | 675.89 | 675.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 669.55 | 674.62 | 674.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 669.55 | 674.62 | 674.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 666.55 | 673.00 | 673.78 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 681.20 | 673.93 | 673.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 11:15:00 | 681.95 | 675.53 | 674.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 12:15:00 | 692.50 | 693.47 | 689.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:45:00 | 690.30 | 693.47 | 689.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 690.45 | 693.69 | 690.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 12:30:00 | 696.25 | 693.85 | 691.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 690.45 | 691.58 | 691.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 690.45 | 691.58 | 691.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 675.20 | 687.40 | 689.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 675.10 | 671.51 | 678.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 674.10 | 672.03 | 677.80 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 685.00 | 679.92 | 679.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 691.30 | 683.79 | 681.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 684.75 | 685.21 | 683.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 708.75 | 715.53 | 710.93 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 707.25 | 711.66 | 711.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 715.55 | 711.91 | 711.68 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 708.00 | 710.86 | 711.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 15:15:00 | 706.15 | 709.92 | 710.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 706.70 | 705.02 | 707.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 14:00:00 | 706.70 | 705.02 | 707.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 704.50 | 704.92 | 707.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 706.25 | 704.92 | 707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 704.35 | 704.45 | 706.64 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 710.55 | 706.33 | 706.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 733.90 | 716.02 | 711.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 731.95 | 737.02 | 728.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:00:00 | 731.95 | 737.02 | 728.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 729.60 | 734.43 | 728.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 729.60 | 734.43 | 728.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 727.90 | 733.12 | 728.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 729.25 | 733.12 | 728.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 726.50 | 731.80 | 728.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 733.75 | 731.80 | 728.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 728.60 | 734.65 | 733.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 721.15 | 731.95 | 732.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 721.15 | 731.95 | 732.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 718.05 | 722.61 | 724.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 716.45 | 715.78 | 719.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 730.50 | 715.78 | 719.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 734.25 | 719.47 | 720.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 735.60 | 719.47 | 720.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 733.65 | 722.31 | 721.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 737.70 | 731.68 | 727.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 735.00 | 735.34 | 731.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 735.00 | 735.34 | 731.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 735.50 | 740.30 | 737.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 735.50 | 740.30 | 737.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 737.25 | 739.69 | 737.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 747.60 | 741.31 | 738.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 776.10 | 780.61 | 780.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 776.10 | 780.61 | 780.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 772.55 | 777.74 | 779.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 761.00 | 759.95 | 765.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 763.45 | 759.95 | 765.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 765.00 | 760.93 | 764.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 765.00 | 760.93 | 764.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 762.35 | 761.21 | 764.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 761.05 | 761.21 | 764.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 761.60 | 761.18 | 764.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 760.75 | 761.15 | 762.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:45:00 | 761.10 | 761.34 | 762.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 758.45 | 760.71 | 761.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 751.70 | 757.18 | 759.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 751.90 | 756.27 | 757.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 760.15 | 758.20 | 757.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 760.15 | 758.20 | 757.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 762.95 | 759.15 | 758.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 767.10 | 764.54 | 762.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 766.25 | 764.59 | 762.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 759.85 | 763.36 | 762.33 | SL hit (close<static) qty=1.00 sl=762.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 754.50 | 761.06 | 761.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 751.00 | 756.98 | 759.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 753.45 | 751.68 | 755.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 753.45 | 751.68 | 755.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 757.80 | 752.96 | 755.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 760.90 | 752.96 | 755.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 763.70 | 755.11 | 755.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 763.70 | 755.11 | 755.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 767.90 | 757.67 | 757.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 770.00 | 760.13 | 758.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 769.60 | 771.97 | 768.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 768.40 | 771.25 | 768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 766.85 | 771.25 | 768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 768.35 | 770.67 | 768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 767.75 | 770.67 | 768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 768.80 | 770.30 | 768.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:15:00 | 767.50 | 770.30 | 768.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 766.80 | 769.60 | 768.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 766.80 | 769.60 | 768.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 765.45 | 768.77 | 767.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 766.70 | 768.77 | 767.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 760.60 | 766.48 | 767.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 756.35 | 764.45 | 766.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 772.10 | 763.86 | 764.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 778.45 | 766.78 | 766.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 779.90 | 769.40 | 767.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 776.90 | 775.25 | 772.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 776.95 | 776.32 | 773.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 780.60 | 775.46 | 773.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 800.30 | 805.21 | 805.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 800.30 | 805.21 | 805.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 795.20 | 802.46 | 804.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 786.55 | 786.45 | 791.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 786.55 | 786.45 | 791.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 786.00 | 784.98 | 788.55 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 792.15 | 789.07 | 788.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 795.15 | 791.01 | 789.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 786.10 | 788.83 | 789.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 785.10 | 787.62 | 788.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 763.25 | 762.38 | 768.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 763.25 | 762.38 | 768.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 767.50 | 764.38 | 767.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:45:00 | 762.00 | 763.39 | 766.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 760.55 | 760.29 | 763.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 754.90 | 750.20 | 750.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 754.90 | 750.20 | 750.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 760.10 | 753.22 | 751.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 759.70 | 759.82 | 756.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:45:00 | 760.60 | 759.82 | 756.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 759.70 | 761.46 | 759.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 761.30 | 761.46 | 759.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 761.50 | 761.47 | 759.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 763.95 | 761.96 | 760.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 757.40 | 761.05 | 759.80 | SL hit (close<static) qty=1.00 sl=759.05 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 753.00 | 758.98 | 759.05 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 12:15:00 | 764.70 | 759.79 | 759.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 767.85 | 763.50 | 761.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 761.05 | 763.19 | 761.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 758.95 | 762.34 | 761.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 758.95 | 762.34 | 761.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 761.40 | 762.06 | 761.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 760.35 | 762.06 | 761.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 763.10 | 762.27 | 761.62 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 14:15:00 | 756.00 | 760.97 | 761.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 15:15:00 | 755.15 | 759.81 | 760.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 760.45 | 759.94 | 760.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 752.25 | 758.40 | 759.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 749.95 | 756.34 | 758.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:45:00 | 749.20 | 755.52 | 756.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 749.25 | 752.53 | 755.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 13:15:00 | 750.75 | 743.52 | 743.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 750.75 | 743.52 | 743.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 09:15:00 | 754.90 | 748.34 | 745.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 755.05 | 757.57 | 754.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:00:00 | 755.05 | 757.57 | 754.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 758.00 | 757.66 | 754.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:45:00 | 759.65 | 758.13 | 755.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 789.90 | 792.98 | 793.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 789.90 | 792.98 | 793.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 788.35 | 791.43 | 792.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 779.00 | 778.11 | 782.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 779.00 | 778.11 | 782.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 786.65 | 779.82 | 782.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 786.65 | 779.82 | 782.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 783.25 | 780.51 | 782.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 780.80 | 780.37 | 782.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 779.95 | 778.26 | 778.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 782.30 | 779.07 | 778.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 782.30 | 779.07 | 778.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 794.40 | 782.61 | 780.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 783.80 | 784.01 | 781.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 783.80 | 784.01 | 781.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 778.70 | 782.95 | 781.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 778.70 | 782.95 | 781.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 784.35 | 783.23 | 781.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 780.00 | 783.23 | 781.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 783.70 | 783.32 | 781.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 777.50 | 783.32 | 781.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 776.10 | 781.88 | 781.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 775.85 | 781.88 | 781.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 774.70 | 780.44 | 780.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 763.55 | 776.15 | 778.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 15:15:00 | 756.90 | 756.54 | 760.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 757.15 | 756.54 | 760.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 757.80 | 756.21 | 759.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 759.05 | 756.21 | 759.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 758.90 | 756.75 | 759.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 758.90 | 756.75 | 759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 757.30 | 756.86 | 759.20 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 771.15 | 761.01 | 760.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 774.00 | 763.61 | 761.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 770.95 | 773.40 | 770.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 775.20 | 773.76 | 771.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 777.60 | 773.76 | 771.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 775.90 | 776.36 | 774.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 775.65 | 777.27 | 775.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 773.75 | 774.90 | 775.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 773.75 | 774.90 | 775.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 770.55 | 773.72 | 774.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 776.00 | 770.37 | 771.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 783.65 | 773.03 | 773.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 788.30 | 776.08 | 774.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 12:15:00 | 782.65 | 784.50 | 780.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 12:45:00 | 782.95 | 784.50 | 780.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 780.65 | 783.59 | 780.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 780.65 | 783.59 | 780.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 785.00 | 783.87 | 781.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 794.00 | 783.87 | 781.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:45:00 | 785.50 | 785.56 | 782.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:30:00 | 785.45 | 786.18 | 783.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 780.70 | 785.74 | 784.19 | SL hit (close<static) qty=1.00 sl=780.75 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 774.95 | 782.19 | 782.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 769.05 | 776.33 | 779.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 768.00 | 766.36 | 770.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:15:00 | 768.15 | 766.36 | 770.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 770.05 | 767.09 | 770.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 771.30 | 767.09 | 770.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 769.50 | 767.58 | 770.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 771.55 | 767.58 | 770.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 769.00 | 767.86 | 770.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:30:00 | 766.55 | 767.69 | 769.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:45:00 | 766.20 | 765.96 | 768.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 765.50 | 760.34 | 760.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 765.50 | 760.34 | 760.06 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 758.05 | 759.68 | 759.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 755.00 | 758.71 | 759.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 763.30 | 759.31 | 759.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 763.30 | 760.11 | 759.77 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 756.85 | 759.55 | 759.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 755.70 | 758.78 | 759.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 15:15:00 | 750.30 | 749.67 | 753.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:15:00 | 743.20 | 749.67 | 753.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 750.80 | 748.09 | 751.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 750.80 | 748.09 | 751.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 752.55 | 748.98 | 751.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 752.55 | 748.98 | 751.42 | SL hit (close>ema400) qty=1.00 sl=751.42 alert=retest1 |

### Cycle 99 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 757.00 | 752.67 | 752.60 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 746.85 | 751.86 | 752.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 745.25 | 748.81 | 750.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 748.85 | 748.82 | 750.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 748.85 | 748.82 | 750.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 757.65 | 746.54 | 747.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 758.35 | 746.54 | 747.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 757.60 | 748.75 | 748.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 759.65 | 753.19 | 750.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 13:15:00 | 743.05 | 749.22 | 749.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 740.95 | 743.79 | 745.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 745.00 | 743.48 | 744.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 743.95 | 743.57 | 744.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 750.20 | 743.57 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 748.70 | 744.60 | 745.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 750.50 | 744.60 | 745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 745.75 | 744.83 | 745.16 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 746.65 | 745.39 | 745.37 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 743.05 | 744.96 | 745.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 738.30 | 743.19 | 744.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 733.85 | 738.20 | 740.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:30:00 | 734.90 | 737.50 | 739.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 745.10 | 739.12 | 739.74 | SL hit (close>static) qty=1.00 sl=743.55 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 744.40 | 740.18 | 740.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 748.25 | 743.49 | 741.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 749.50 | 755.04 | 750.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 750.50 | 754.14 | 750.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 750.00 | 754.14 | 750.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 745.65 | 752.44 | 750.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 746.00 | 752.44 | 750.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 747.05 | 751.36 | 749.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 747.05 | 751.36 | 749.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 743.50 | 747.87 | 748.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 741.60 | 746.62 | 747.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 736.35 | 736.23 | 740.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 15:00:00 | 735.20 | 736.15 | 739.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 741.90 | 737.43 | 739.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 741.90 | 737.43 | 739.50 | SL hit (close>ema400) qty=1.00 sl=739.50 alert=retest1 |

### Cycle 107 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 745.65 | 741.26 | 740.91 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 738.90 | 740.81 | 740.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 735.10 | 739.20 | 740.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 739.20 | 738.91 | 739.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 737.20 | 738.57 | 739.55 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 745.50 | 740.71 | 740.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 749.50 | 742.47 | 741.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 750.10 | 751.47 | 747.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 750.10 | 751.47 | 747.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 752.80 | 751.50 | 748.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 750.40 | 751.50 | 748.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 771.50 | 779.41 | 774.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 771.50 | 779.41 | 774.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 775.60 | 778.65 | 774.27 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 768.80 | 772.66 | 772.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 11:15:00 | 766.35 | 771.40 | 772.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 770.25 | 770.25 | 771.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 763.75 | 759.20 | 761.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 763.75 | 759.20 | 761.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 761.05 | 759.57 | 761.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:15:00 | 765.00 | 759.57 | 761.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 765.00 | 760.65 | 761.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 765.00 | 761.26 | 761.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 763.30 | 761.67 | 761.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:15:00 | 763.50 | 761.67 | 761.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 764.00 | 762.14 | 762.10 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 761.00 | 762.27 | 762.44 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 768.80 | 763.49 | 762.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 780.30 | 767.56 | 764.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 779.20 | 781.00 | 774.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:45:00 | 778.85 | 781.00 | 774.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 777.40 | 779.58 | 775.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 777.40 | 779.58 | 775.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 771.40 | 777.49 | 775.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 772.75 | 777.49 | 775.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 769.40 | 775.87 | 775.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 769.90 | 775.87 | 775.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 766.10 | 773.92 | 774.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 764.50 | 772.03 | 773.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 767.55 | 764.69 | 767.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 765.00 | 764.75 | 767.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 760.45 | 764.75 | 767.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 761.15 | 757.10 | 756.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 761.15 | 757.10 | 756.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 767.55 | 759.19 | 757.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 770.20 | 770.51 | 766.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:30:00 | 769.70 | 770.51 | 766.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 764.20 | 769.25 | 765.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 764.20 | 769.25 | 765.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 762.30 | 767.86 | 765.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 756.95 | 767.86 | 765.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 756.45 | 763.48 | 763.83 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 769.85 | 764.29 | 763.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 771.70 | 765.77 | 764.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 764.95 | 768.12 | 766.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 768.45 | 768.19 | 766.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 769.75 | 768.37 | 766.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 770.20 | 772.15 | 771.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 14:15:00 | 765.45 | 771.45 | 771.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 765.45 | 771.45 | 771.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 762.80 | 769.72 | 770.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 756.40 | 755.51 | 760.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 756.20 | 755.51 | 760.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 756.25 | 756.23 | 758.84 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 764.75 | 759.71 | 759.70 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 758.35 | 761.82 | 761.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 755.30 | 760.52 | 761.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 749.40 | 748.97 | 753.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 749.40 | 748.97 | 753.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 746.15 | 742.95 | 746.28 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 750.00 | 747.91 | 747.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 15:15:00 | 751.95 | 748.72 | 748.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 748.40 | 749.35 | 748.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 748.75 | 749.23 | 748.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:30:00 | 750.25 | 749.40 | 748.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 754.25 | 767.13 | 767.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 754.25 | 767.13 | 767.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 752.75 | 758.28 | 761.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 761.05 | 754.90 | 758.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 756.10 | 755.14 | 758.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 754.35 | 755.14 | 758.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:45:00 | 755.35 | 755.57 | 757.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 754.40 | 755.33 | 757.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 716.63 | 723.99 | 727.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 717.58 | 723.99 | 727.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 716.68 | 723.99 | 727.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 720.05 | 717.40 | 721.66 | SL hit (close>ema200) qty=0.50 sl=717.40 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 725.95 | 722.00 | 721.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 728.15 | 723.81 | 722.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 727.05 | 727.19 | 725.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 15:15:00 | 725.55 | 727.19 | 725.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 725.55 | 726.86 | 725.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 722.15 | 726.86 | 725.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 723.80 | 726.25 | 725.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 721.45 | 726.25 | 725.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 726.00 | 726.03 | 725.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 726.00 | 726.03 | 725.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 729.35 | 726.70 | 725.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:45:00 | 734.90 | 728.17 | 726.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 734.05 | 728.53 | 726.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 719.25 | 727.00 | 726.84 | SL hit (close<static) qty=1.00 sl=724.10 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 718.80 | 725.52 | 726.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 714.65 | 723.35 | 725.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 717.85 | 717.28 | 720.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 717.85 | 717.28 | 720.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 730.00 | 719.94 | 721.31 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 724.25 | 722.21 | 722.14 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 720.10 | 721.83 | 721.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 719.15 | 721.00 | 721.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 723.90 | 721.17 | 721.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 726.10 | 722.15 | 721.88 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 715.10 | 720.66 | 721.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 710.65 | 716.91 | 718.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 708.75 | 707.64 | 712.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 708.75 | 707.64 | 712.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 706.80 | 707.44 | 709.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 705.50 | 707.25 | 709.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:30:00 | 705.70 | 706.85 | 709.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:00:00 | 705.50 | 706.85 | 709.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 706.30 | 701.60 | 701.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 706.30 | 701.60 | 701.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 708.45 | 705.25 | 703.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 705.00 | 705.38 | 703.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:15:00 | 707.55 | 705.38 | 703.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 706.65 | 706.27 | 704.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:15:00 | 742.93 | 732.11 | 727.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:15:00 | 741.98 | 732.11 | 727.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 738.15 | 739.83 | 734.60 | SL hit (close<ema200) qty=0.50 sl=739.83 alert=retest1 |

### Cycle 130 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 733.25 | 736.05 | 736.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 725.85 | 734.01 | 735.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 681.90 | 680.72 | 691.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 685.35 | 680.72 | 691.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 686.20 | 681.81 | 691.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 678.80 | 681.81 | 691.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 644.86 | 669.19 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 659.30 | 657.96 | 664.91 | SL hit (close>ema200) qty=0.50 sl=657.96 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 642.90 | 635.28 | 634.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 645.70 | 641.08 | 638.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 631.90 | 638.54 | 638.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 632.65 | 638.54 | 638.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 633.85 | 637.32 | 637.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 633.85 | 637.32 | 637.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 628.85 | 634.44 | 636.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 605.20 | 599.21 | 608.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 605.20 | 599.21 | 608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 612.30 | 603.60 | 607.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 612.30 | 603.60 | 607.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 614.90 | 605.86 | 608.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 614.90 | 605.86 | 608.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 612.95 | 609.92 | 609.87 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 603.25 | 609.08 | 609.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 601.15 | 606.90 | 608.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 608.15 | 607.01 | 608.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 609.70 | 607.55 | 608.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 609.70 | 607.55 | 608.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 610.40 | 608.12 | 608.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 610.40 | 608.12 | 608.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 610.90 | 608.67 | 608.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 600.05 | 608.67 | 608.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 14:15:00 | 570.05 | 582.91 | 592.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 568.20 | 567.33 | 577.83 | SL hit (close>ema200) qty=0.50 sl=567.33 alert=retest2 |

### Cycle 135 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 580.65 | 576.72 | 576.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 582.05 | 577.79 | 576.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 589.05 | 595.45 | 589.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 588.45 | 593.14 | 589.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 588.45 | 593.14 | 589.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 586.30 | 591.77 | 589.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 585.75 | 591.77 | 589.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 593.00 | 592.00 | 589.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 596.60 | 592.00 | 589.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 596.20 | 592.04 | 590.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 614.00 | 624.00 | 625.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 614.00 | 624.00 | 625.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 609.75 | 617.55 | 621.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 613.50 | 611.35 | 615.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 613.85 | 612.12 | 614.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 615.10 | 612.12 | 614.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 614.30 | 612.56 | 614.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 614.30 | 612.56 | 614.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 614.10 | 612.87 | 614.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 613.00 | 612.12 | 614.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 602.25 | 597.27 | 596.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 602.25 | 597.27 | 596.67 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 579.70 | 593.21 | 594.99 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 592.15 | 590.73 | 590.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 600.00 | 594.52 | 592.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 622.00 | 622.72 | 616.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 622.00 | 622.72 | 616.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 566.70 | 2024-05-27 14:15:00 | 564.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-05-27 10:15:00 | 566.75 | 2024-05-27 14:15:00 | 564.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-28 09:15:00 | 567.00 | 2024-05-29 13:15:00 | 563.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-29 10:30:00 | 566.05 | 2024-05-29 13:15:00 | 563.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-06-04 11:00:00 | 533.95 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-06-04 11:30:00 | 536.20 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-06-04 14:15:00 | 535.10 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-06-04 14:45:00 | 534.85 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-06-13 11:15:00 | 592.30 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-13 14:15:00 | 592.00 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-06-14 09:15:00 | 596.30 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-06-28 09:15:00 | 595.35 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-28 10:00:00 | 596.45 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-06-28 14:45:00 | 595.70 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-07-01 10:30:00 | 596.55 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-07-02 11:45:00 | 595.35 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-15 12:00:00 | 639.00 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-07-16 09:15:00 | 647.10 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-07-16 10:30:00 | 639.30 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-07-22 12:30:00 | 636.25 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-22 14:00:00 | 636.25 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-23 09:45:00 | 632.85 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-07-23 12:15:00 | 624.20 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-07-30 11:15:00 | 699.60 | 2024-08-05 09:15:00 | 698.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-31 09:45:00 | 698.95 | 2024-08-05 09:15:00 | 698.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-08-08 09:15:00 | 692.00 | 2024-08-08 11:15:00 | 703.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-08-08 10:30:00 | 691.90 | 2024-08-08 11:15:00 | 703.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-08-19 09:15:00 | 680.70 | 2024-08-20 09:15:00 | 696.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-08-19 11:00:00 | 681.05 | 2024-08-20 09:15:00 | 696.95 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-08-27 09:45:00 | 728.80 | 2024-09-06 10:15:00 | 743.55 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2024-08-27 10:45:00 | 729.15 | 2024-09-06 10:15:00 | 743.55 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2024-09-13 09:30:00 | 708.70 | 2024-09-19 10:15:00 | 706.05 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-09-26 10:15:00 | 726.95 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-26 11:00:00 | 727.05 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-26 11:30:00 | 726.65 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-26 12:30:00 | 727.70 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-04 13:30:00 | 705.60 | 2024-10-07 09:15:00 | 716.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-10-07 12:15:00 | 707.00 | 2024-10-09 09:15:00 | 718.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-10-16 09:15:00 | 727.00 | 2024-10-16 09:15:00 | 723.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-10-29 09:15:00 | 716.25 | 2024-10-29 10:15:00 | 723.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-05 15:15:00 | 715.50 | 2024-11-08 09:15:00 | 717.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-11-06 09:30:00 | 711.70 | 2024-11-08 09:15:00 | 717.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-11-11 13:45:00 | 707.35 | 2024-11-19 11:15:00 | 671.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:15:00 | 707.75 | 2024-11-19 11:15:00 | 672.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 706.90 | 2024-11-19 11:15:00 | 671.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:45:00 | 707.35 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-11-12 10:15:00 | 707.75 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2024-11-12 11:45:00 | 706.90 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2024-11-25 14:30:00 | 689.60 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-11-26 10:30:00 | 690.95 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-11-26 12:00:00 | 689.55 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-11-28 10:30:00 | 681.25 | 2024-11-28 13:15:00 | 647.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 10:30:00 | 681.25 | 2024-11-29 14:15:00 | 658.00 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-11-28 12:45:00 | 669.90 | 2024-12-03 09:15:00 | 636.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 12:45:00 | 669.90 | 2024-12-04 09:15:00 | 645.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2024-12-20 09:15:00 | 619.65 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-12-20 12:15:00 | 623.00 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-20 14:00:00 | 623.00 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-20 14:45:00 | 622.65 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-23 09:15:00 | 617.20 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-12-23 13:00:00 | 620.85 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-23 15:15:00 | 620.50 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-01-10 14:15:00 | 610.40 | 2025-01-16 09:15:00 | 653.10 | STOP_HIT | 1.00 | -7.00% |
| SELL | retest2 | 2025-01-23 14:45:00 | 621.35 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-01-24 12:30:00 | 620.85 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-01-24 14:00:00 | 621.05 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-02-14 09:15:00 | 632.75 | 2025-02-14 09:15:00 | 627.75 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-18 11:00:00 | 612.60 | 2025-02-19 15:15:00 | 623.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-02-25 15:15:00 | 620.00 | 2025-02-25 15:15:00 | 620.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-03-06 13:00:00 | 617.85 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-03-06 14:15:00 | 617.60 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-03-07 09:15:00 | 631.55 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-04-03 12:30:00 | 696.25 | 2025-04-04 14:15:00 | 690.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-05 09:15:00 | 733.75 | 2025-05-06 12:15:00 | 721.15 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-05-06 11:45:00 | 728.60 | 2025-05-06 12:15:00 | 721.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-15 13:00:00 | 747.60 | 2025-05-30 13:15:00 | 776.10 | STOP_HIT | 1.00 | 3.81% |
| SELL | retest2 | 2025-06-04 13:15:00 | 761.05 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-06-04 13:45:00 | 761.60 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-06-05 13:30:00 | 760.75 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-06-05 14:45:00 | 761.10 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-06-09 09:15:00 | 751.70 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-10 09:15:00 | 751.90 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-12 09:15:00 | 767.10 | 2025-06-12 11:15:00 | 759.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-12 10:15:00 | 766.25 | 2025-06-12 11:15:00 | 759.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-23 11:45:00 | 776.90 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-06-23 14:15:00 | 776.95 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-06-24 09:15:00 | 780.60 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-07-15 11:45:00 | 762.00 | 2025-07-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-07-16 10:30:00 | 760.55 | 2025-07-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-07-24 14:00:00 | 763.95 | 2025-07-24 14:15:00 | 757.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-31 10:15:00 | 749.95 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-08-01 09:45:00 | 749.20 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-08-01 11:30:00 | 749.25 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-08-11 12:45:00 | 759.65 | 2025-08-22 11:15:00 | 789.90 | STOP_HIT | 1.00 | 3.98% |
| SELL | retest2 | 2025-08-28 12:30:00 | 780.80 | 2025-09-01 13:15:00 | 782.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-09-01 12:30:00 | 779.95 | 2025-09-01 13:15:00 | 782.30 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-09-12 11:15:00 | 777.60 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-15 10:30:00 | 775.90 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-16 09:45:00 | 775.65 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-09-22 09:15:00 | 794.00 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-09-22 11:45:00 | 785.50 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-22 12:30:00 | 785.45 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-26 13:30:00 | 766.55 | 2025-10-01 15:15:00 | 765.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-29 09:45:00 | 766.20 | 2025-10-01 15:15:00 | 765.50 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest1 | 2025-10-09 09:15:00 | 743.20 | 2025-10-09 12:15:00 | 752.55 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-27 12:15:00 | 733.85 | 2025-10-28 10:15:00 | 745.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-27 13:30:00 | 734.90 | 2025-10-28 10:15:00 | 745.10 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest1 | 2025-11-03 15:00:00 | 735.20 | 2025-11-04 09:15:00 | 741.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-02 09:15:00 | 760.45 | 2025-12-05 10:15:00 | 761.15 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-12-11 12:15:00 | 769.75 | 2025-12-16 14:15:00 | 765.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-15 11:30:00 | 770.20 | 2025-12-16 14:15:00 | 765.45 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-01 14:30:00 | 750.25 | 2026-01-08 10:15:00 | 754.25 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2026-01-12 11:15:00 | 754.35 | 2026-01-23 09:15:00 | 716.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 13:45:00 | 755.35 | 2026-01-23 09:15:00 | 717.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 15:00:00 | 754.40 | 2026-01-23 09:15:00 | 716.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 11:15:00 | 754.35 | 2026-01-27 10:15:00 | 720.05 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2026-01-12 13:45:00 | 755.35 | 2026-01-27 10:15:00 | 720.05 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2026-01-12 15:00:00 | 754.40 | 2026-01-27 10:15:00 | 720.05 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2026-01-30 13:45:00 | 734.90 | 2026-02-01 12:15:00 | 719.25 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-01-30 14:45:00 | 734.05 | 2026-02-01 12:15:00 | 719.25 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-10 11:15:00 | 705.50 | 2026-02-16 14:15:00 | 706.30 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-02-10 12:30:00 | 705.70 | 2026-02-16 14:15:00 | 706.30 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-02-10 13:00:00 | 705.50 | 2026-02-16 14:15:00 | 706.30 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2026-02-18 09:15:00 | 707.55 | 2026-02-23 09:15:00 | 742.93 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-18 10:45:00 | 706.65 | 2026-02-23 09:15:00 | 741.98 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-18 09:15:00 | 707.55 | 2026-02-24 09:15:00 | 738.15 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest1 | 2026-02-18 10:45:00 | 706.65 | 2026-02-24 09:15:00 | 738.15 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2026-02-26 09:15:00 | 741.65 | 2026-02-26 13:15:00 | 734.55 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-03-06 09:15:00 | 678.80 | 2026-03-09 09:15:00 | 644.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 678.80 | 2026-03-10 12:15:00 | 659.30 | STOP_HIT | 0.50 | 2.87% |
| BUY | retest2 | 2026-03-19 10:30:00 | 631.90 | 2026-03-19 13:15:00 | 633.85 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-03-19 11:00:00 | 632.65 | 2026-03-19 13:15:00 | 633.85 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-03-30 09:15:00 | 600.05 | 2026-04-01 14:15:00 | 570.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-30 09:15:00 | 600.05 | 2026-04-02 14:15:00 | 568.20 | STOP_HIT | 0.50 | 5.31% |
| BUY | retest2 | 2026-04-10 09:15:00 | 596.60 | 2026-04-17 11:15:00 | 614.00 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2026-04-10 10:15:00 | 596.20 | 2026-04-17 11:15:00 | 614.00 | STOP_HIT | 1.00 | 2.99% |
| SELL | retest2 | 2026-04-22 09:30:00 | 613.00 | 2026-04-29 12:15:00 | 602.25 | STOP_HIT | 1.00 | 1.75% |
