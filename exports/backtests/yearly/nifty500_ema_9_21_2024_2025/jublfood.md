# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 473.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 143 |
| ALERT1 | 105 |
| ALERT2 | 104 |
| ALERT2_SKIP | 59 |
| ALERT3 | 285 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 134 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 98
- **Target hits / Stop hits / Partials:** 8 / 131 / 15
- **Avg / median % per leg:** 0.59% / -0.68%
- **Sum % (uncompounded):** 90.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 17 | 32.1% | 6 | 47 | 0 | 0.78% | 41.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.56% | -4.7% |
| BUY @ 3rd Alert (retest2) | 50 | 17 | 34.0% | 6 | 44 | 0 | 0.92% | 46.0% |
| SELL (all) | 101 | 39 | 38.6% | 2 | 84 | 15 | 0.48% | 48.8% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.05% | 16.2% |
| SELL @ 3rd Alert (retest2) | 97 | 35 | 36.1% | 2 | 82 | 13 | 0.34% | 32.6% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.64% | 11.5% |
| retest2 (combined) | 147 | 52 | 35.4% | 8 | 126 | 13 | 0.54% | 78.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 471.90 | 465.47 | 465.45 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 467.80 | 468.42 | 468.46 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 472.10 | 468.93 | 468.68 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 465.75 | 468.91 | 469.07 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 470.50 | 469.23 | 469.20 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 467.40 | 468.86 | 469.04 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 12:15:00 | 475.30 | 470.15 | 469.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 14:15:00 | 480.20 | 473.24 | 471.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 477.25 | 477.95 | 475.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 477.25 | 477.95 | 475.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 477.25 | 477.95 | 475.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 477.25 | 477.95 | 475.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 474.45 | 477.25 | 475.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 469.15 | 477.25 | 475.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 477.55 | 477.31 | 475.71 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 470.35 | 474.85 | 475.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 466.70 | 470.46 | 472.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 481.70 | 472.05 | 472.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 481.70 | 472.05 | 472.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 481.70 | 472.05 | 472.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 481.70 | 472.05 | 472.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 486.50 | 474.94 | 474.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 14:15:00 | 491.90 | 482.92 | 478.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 511.00 | 511.05 | 502.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 09:15:00 | 517.95 | 511.05 | 502.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 503.75 | 511.00 | 506.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 503.75 | 511.00 | 506.69 | SL hit (close<ema400) qty=1.00 sl=506.69 alert=retest1 |

### Cycle 10 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 495.05 | 504.81 | 504.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 14:15:00 | 492.45 | 499.75 | 502.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 500.70 | 499.58 | 501.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 500.70 | 499.58 | 501.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 500.70 | 499.58 | 501.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 495.40 | 498.75 | 501.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 495.50 | 499.02 | 501.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 485.25 | 498.37 | 500.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 492.35 | 496.39 | 498.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 470.63 | 490.61 | 495.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 470.72 | 490.61 | 495.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 498.20 | 491.47 | 495.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 498.20 | 491.47 | 495.14 | SL hit (close>ema200) qty=0.50 sl=491.47 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 520.00 | 498.56 | 497.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 538.20 | 506.49 | 501.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 522.00 | 522.50 | 513.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:15:00 | 519.05 | 522.50 | 513.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 519.95 | 521.19 | 517.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 519.35 | 521.19 | 517.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 516.20 | 520.20 | 517.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:00:00 | 516.20 | 520.20 | 517.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 515.50 | 519.26 | 517.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:45:00 | 515.90 | 519.26 | 517.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 517.75 | 518.96 | 517.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:30:00 | 519.40 | 518.95 | 517.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 519.55 | 518.80 | 517.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:15:00 | 519.20 | 517.84 | 517.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:45:00 | 519.25 | 517.88 | 517.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 522.25 | 518.75 | 517.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 524.50 | 519.91 | 518.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 15:15:00 | 538.00 | 541.30 | 541.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 15:15:00 | 538.00 | 541.30 | 541.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 09:15:00 | 537.05 | 540.45 | 541.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 11:15:00 | 543.20 | 540.81 | 541.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 11:15:00 | 543.20 | 540.81 | 541.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 543.20 | 540.81 | 541.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:45:00 | 544.05 | 540.81 | 541.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 12:15:00 | 547.90 | 542.23 | 541.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 550.25 | 543.83 | 542.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 546.10 | 546.25 | 544.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 546.10 | 546.25 | 544.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 546.10 | 546.25 | 544.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 546.10 | 546.25 | 544.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 553.70 | 547.74 | 545.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 12:15:00 | 556.35 | 548.75 | 545.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 556.50 | 558.17 | 557.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:45:00 | 556.50 | 557.47 | 556.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:30:00 | 557.25 | 557.86 | 557.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 566.20 | 559.53 | 558.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 560.30 | 559.53 | 558.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 552.70 | 558.18 | 557.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 552.70 | 558.18 | 557.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 552.70 | 557.09 | 557.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 552.70 | 557.09 | 557.25 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 561.15 | 557.37 | 557.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 566.80 | 560.08 | 558.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 564.75 | 569.99 | 566.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 564.75 | 569.99 | 566.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 564.75 | 569.99 | 566.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 564.75 | 569.99 | 566.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 570.35 | 570.06 | 566.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 569.80 | 570.06 | 566.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 574.70 | 570.71 | 567.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 576.20 | 571.13 | 568.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 15:15:00 | 565.50 | 569.18 | 569.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 15:15:00 | 565.50 | 569.18 | 569.34 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 571.75 | 569.36 | 569.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 576.50 | 571.01 | 570.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 570.00 | 570.81 | 570.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 570.00 | 570.81 | 570.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 570.00 | 570.81 | 570.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 570.00 | 570.81 | 570.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 566.55 | 569.96 | 569.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 566.55 | 569.96 | 569.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 568.80 | 569.72 | 569.64 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 568.15 | 569.41 | 569.51 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 574.70 | 570.47 | 569.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 583.25 | 574.05 | 572.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 10:15:00 | 573.35 | 576.39 | 574.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 573.35 | 576.39 | 574.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 573.35 | 576.39 | 574.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 573.35 | 576.39 | 574.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 576.90 | 576.50 | 574.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:15:00 | 580.10 | 576.33 | 574.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 579.45 | 578.81 | 576.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:30:00 | 579.30 | 578.58 | 578.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 578.55 | 578.80 | 578.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 582.20 | 580.27 | 579.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 582.20 | 580.27 | 579.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 581.35 | 583.70 | 581.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 582.30 | 583.70 | 581.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 575.70 | 582.10 | 581.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 575.70 | 582.10 | 581.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 570.95 | 579.87 | 580.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 570.95 | 579.87 | 580.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 12:15:00 | 567.65 | 577.42 | 579.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 09:15:00 | 575.25 | 573.35 | 576.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 575.25 | 573.35 | 576.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 575.25 | 573.35 | 576.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 575.25 | 573.35 | 576.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 574.25 | 573.53 | 576.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 579.10 | 573.53 | 576.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 570.90 | 566.95 | 570.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 570.90 | 566.95 | 570.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 573.25 | 568.21 | 570.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 575.25 | 568.21 | 570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 569.35 | 568.44 | 570.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 564.80 | 568.33 | 570.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:45:00 | 566.55 | 567.82 | 569.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 14:15:00 | 582.50 | 567.98 | 567.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 582.50 | 567.98 | 567.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 591.75 | 581.96 | 578.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 586.35 | 588.33 | 585.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 14:00:00 | 586.35 | 588.33 | 585.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 580.10 | 586.69 | 584.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 580.10 | 586.69 | 584.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 579.35 | 585.22 | 584.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 592.80 | 585.22 | 584.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 588.55 | 598.36 | 598.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 588.55 | 598.36 | 598.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 582.45 | 595.18 | 597.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 601.00 | 592.92 | 595.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 601.00 | 592.92 | 595.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 601.00 | 592.92 | 595.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 601.00 | 592.92 | 595.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 601.30 | 594.59 | 595.66 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 604.20 | 596.52 | 596.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 604.85 | 598.18 | 597.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 596.30 | 598.43 | 597.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 596.30 | 598.43 | 597.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 596.30 | 598.43 | 597.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 596.30 | 598.43 | 597.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 597.10 | 598.17 | 597.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 603.55 | 598.17 | 597.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 13:30:00 | 600.15 | 602.46 | 602.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 14:15:00 | 596.45 | 601.26 | 601.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 596.45 | 601.26 | 601.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 595.60 | 600.13 | 601.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 601.60 | 600.42 | 601.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 601.60 | 600.42 | 601.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 601.60 | 600.42 | 601.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 599.05 | 600.15 | 600.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 599.15 | 599.61 | 600.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:30:00 | 598.40 | 599.56 | 600.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 599.05 | 599.41 | 600.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 615.40 | 602.44 | 601.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 615.40 | 602.44 | 601.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 638.90 | 609.73 | 604.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 642.15 | 642.78 | 633.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 09:30:00 | 642.40 | 642.78 | 633.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 639.05 | 641.42 | 635.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:30:00 | 636.50 | 641.42 | 635.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 635.25 | 640.19 | 635.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 635.25 | 640.19 | 635.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 636.70 | 639.49 | 635.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 645.45 | 639.39 | 635.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:45:00 | 641.25 | 639.89 | 636.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 11:15:00 | 640.40 | 639.28 | 636.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 630.55 | 638.60 | 637.81 | SL hit (close<static) qty=1.00 sl=633.90 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 630.45 | 636.97 | 637.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 628.75 | 632.70 | 634.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 637.90 | 631.27 | 632.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 637.90 | 631.27 | 632.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 637.90 | 631.27 | 632.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 637.20 | 631.27 | 632.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 645.35 | 634.08 | 633.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 12:15:00 | 652.50 | 639.27 | 636.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 650.40 | 653.47 | 648.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 650.40 | 653.47 | 648.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 653.70 | 651.87 | 649.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:30:00 | 657.35 | 652.92 | 650.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:30:00 | 656.35 | 653.58 | 650.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:30:00 | 656.85 | 654.53 | 651.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 658.45 | 660.07 | 658.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 656.45 | 659.35 | 657.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 656.85 | 659.35 | 657.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 652.05 | 657.89 | 657.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 652.65 | 656.84 | 656.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 652.65 | 656.84 | 656.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 650.80 | 655.05 | 656.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 654.30 | 653.59 | 654.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 654.30 | 653.59 | 654.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 654.30 | 653.59 | 654.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 652.75 | 653.59 | 654.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 654.70 | 653.81 | 654.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 654.70 | 653.81 | 654.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 659.75 | 655.00 | 655.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 659.00 | 655.00 | 655.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 657.10 | 655.42 | 655.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 656.10 | 655.42 | 655.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 15:15:00 | 653.80 | 651.83 | 651.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 653.80 | 651.83 | 651.76 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 649.50 | 651.65 | 651.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 11:15:00 | 646.60 | 648.93 | 650.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 646.85 | 645.56 | 647.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:00:00 | 646.85 | 645.56 | 647.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 648.80 | 646.21 | 647.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 648.80 | 646.21 | 647.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 649.95 | 646.96 | 647.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 649.85 | 646.96 | 647.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 647.30 | 647.03 | 647.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 637.55 | 647.00 | 647.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 643.15 | 645.04 | 646.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 654.80 | 644.92 | 645.33 | SL hit (close>static) qty=1.00 sl=650.10 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 660.60 | 648.05 | 646.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 664.50 | 653.07 | 649.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 10:15:00 | 659.10 | 662.61 | 659.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 10:15:00 | 659.10 | 662.61 | 659.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 659.10 | 662.61 | 659.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 659.10 | 662.61 | 659.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 666.05 | 663.30 | 659.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:15:00 | 667.80 | 663.30 | 659.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 668.05 | 669.82 | 669.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 663.40 | 668.53 | 669.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 663.40 | 668.53 | 669.23 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 676.05 | 670.04 | 669.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 12:15:00 | 679.10 | 672.70 | 671.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 15:15:00 | 674.25 | 674.49 | 672.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 680.10 | 675.13 | 672.98 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 678.05 | 675.19 | 673.20 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 673.90 | 674.93 | 673.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:45:00 | 672.55 | 674.93 | 673.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 672.45 | 674.43 | 673.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 672.45 | 674.43 | 673.19 | SL hit (close<ema400) qty=1.00 sl=673.19 alert=retest1 |

### Cycle 34 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 696.75 | 702.46 | 702.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 687.00 | 698.33 | 700.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 694.95 | 693.14 | 696.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 694.95 | 693.14 | 696.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 694.95 | 693.14 | 696.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 694.95 | 693.14 | 696.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 695.75 | 693.90 | 696.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 697.40 | 693.90 | 696.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 693.40 | 693.80 | 696.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:30:00 | 696.50 | 693.80 | 696.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 694.30 | 693.90 | 696.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:45:00 | 696.20 | 693.90 | 696.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 664.00 | 679.26 | 685.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 659.10 | 679.26 | 685.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:30:00 | 659.80 | 669.15 | 679.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 626.14 | 640.59 | 653.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 626.81 | 640.59 | 653.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 624.65 | 619.62 | 625.08 | SL hit (close>ema200) qty=0.50 sl=619.62 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 629.55 | 625.76 | 625.51 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 619.20 | 625.06 | 625.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 615.90 | 623.23 | 624.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 622.50 | 619.79 | 621.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 622.50 | 619.79 | 621.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 622.50 | 619.79 | 621.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 622.50 | 619.79 | 621.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 627.20 | 621.27 | 622.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 627.20 | 621.27 | 622.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 624.50 | 621.92 | 622.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 622.80 | 621.92 | 622.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 626.20 | 623.07 | 622.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 626.20 | 623.07 | 622.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 15:15:00 | 627.95 | 624.05 | 623.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 622.00 | 624.14 | 623.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 10:15:00 | 622.00 | 624.14 | 623.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 622.00 | 624.14 | 623.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 622.00 | 624.14 | 623.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 621.40 | 623.59 | 623.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 619.95 | 623.59 | 623.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 619.60 | 622.80 | 622.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 617.70 | 621.31 | 622.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 11:15:00 | 622.25 | 620.78 | 621.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 11:15:00 | 622.25 | 620.78 | 621.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 622.25 | 620.78 | 621.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 622.25 | 620.78 | 621.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 621.95 | 621.02 | 621.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:30:00 | 621.75 | 621.02 | 621.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 617.45 | 620.30 | 621.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 14:30:00 | 613.75 | 618.74 | 620.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 10:45:00 | 616.65 | 616.75 | 619.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 624.30 | 618.26 | 619.49 | SL hit (close>static) qty=1.00 sl=622.20 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 627.50 | 620.90 | 620.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 631.00 | 623.88 | 622.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 13:15:00 | 619.85 | 626.02 | 624.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 13:15:00 | 619.85 | 626.02 | 624.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 619.85 | 626.02 | 624.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 619.85 | 626.02 | 624.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 621.75 | 625.17 | 623.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:30:00 | 618.85 | 625.17 | 623.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 619.80 | 624.09 | 623.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 622.70 | 624.09 | 623.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 619.15 | 623.10 | 623.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 618.30 | 623.10 | 623.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 617.25 | 621.93 | 622.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 610.95 | 616.82 | 619.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 601.90 | 601.63 | 607.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:00:00 | 599.35 | 601.17 | 606.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 12:45:00 | 594.00 | 599.18 | 604.19 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 12:15:00 | 569.38 | 578.96 | 586.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 13:15:00 | 564.30 | 577.26 | 585.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-29 15:15:00 | 578.20 | 577.10 | 583.77 | SL hit (close>ema200) qty=0.50 sl=577.10 alert=retest1 |

### Cycle 41 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 583.45 | 574.83 | 574.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 589.40 | 579.09 | 576.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 593.95 | 595.87 | 589.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 593.95 | 595.87 | 589.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 587.80 | 594.26 | 589.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 587.80 | 594.26 | 589.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 590.50 | 593.51 | 589.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 585.20 | 593.51 | 589.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 599.55 | 604.50 | 600.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 599.55 | 604.50 | 600.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 602.60 | 604.12 | 600.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:30:00 | 594.70 | 604.12 | 600.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 602.60 | 603.44 | 601.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 647.45 | 603.44 | 601.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 613.20 | 620.43 | 620.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 12:15:00 | 613.20 | 620.43 | 620.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 13:15:00 | 610.15 | 618.37 | 619.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 10:15:00 | 615.35 | 614.63 | 617.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 11:00:00 | 615.35 | 614.63 | 617.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 614.85 | 614.60 | 616.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 616.20 | 614.60 | 616.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 616.95 | 612.61 | 614.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 616.95 | 612.61 | 614.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 618.35 | 613.76 | 615.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 618.35 | 613.76 | 615.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 619.45 | 616.71 | 616.37 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 612.25 | 615.82 | 616.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 610.75 | 614.81 | 615.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 10:15:00 | 616.40 | 614.56 | 615.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 10:15:00 | 616.40 | 614.56 | 615.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 616.40 | 614.56 | 615.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 616.40 | 614.56 | 615.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 610.05 | 613.66 | 614.78 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 629.20 | 615.76 | 614.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 630.20 | 620.21 | 617.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 633.15 | 633.96 | 627.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 633.15 | 633.96 | 627.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 640.80 | 646.85 | 644.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 640.80 | 646.85 | 644.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 642.90 | 646.06 | 644.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:30:00 | 639.15 | 646.06 | 644.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 641.50 | 644.60 | 643.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 648.00 | 645.24 | 644.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 646.50 | 644.96 | 644.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:30:00 | 648.75 | 646.25 | 645.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-11 12:15:00 | 712.80 | 701.60 | 694.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 675.45 | 692.83 | 693.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 674.80 | 689.23 | 691.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 09:15:00 | 682.60 | 681.67 | 686.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 10:00:00 | 682.60 | 681.67 | 686.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 680.50 | 681.31 | 685.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 683.60 | 681.31 | 685.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 687.90 | 682.62 | 685.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:45:00 | 687.40 | 682.62 | 685.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 684.20 | 682.94 | 685.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 14:45:00 | 680.05 | 682.40 | 685.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 690.00 | 685.79 | 685.96 | SL hit (close>static) qty=1.00 sl=689.85 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 690.35 | 686.70 | 686.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 694.95 | 688.37 | 687.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 688.20 | 688.50 | 687.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:15:00 | 688.30 | 688.50 | 687.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 685.25 | 687.85 | 687.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 685.25 | 687.85 | 687.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 682.50 | 686.78 | 686.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 678.05 | 684.45 | 685.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 679.85 | 677.14 | 679.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 679.85 | 677.14 | 679.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 679.85 | 677.14 | 679.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 679.85 | 677.14 | 679.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 680.80 | 677.87 | 679.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 678.20 | 677.90 | 679.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 676.00 | 678.32 | 679.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 677.80 | 677.34 | 679.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 678.50 | 678.19 | 679.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 673.00 | 677.15 | 678.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:30:00 | 678.60 | 677.15 | 678.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 681.00 | 675.95 | 677.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 681.00 | 675.95 | 677.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 689.05 | 678.57 | 678.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 689.05 | 678.57 | 678.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 690.10 | 680.87 | 679.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 10:15:00 | 697.15 | 698.08 | 692.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 10:15:00 | 697.15 | 698.08 | 692.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 697.15 | 698.08 | 692.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 696.65 | 698.08 | 692.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 698.45 | 697.43 | 693.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 14:00:00 | 701.80 | 698.31 | 694.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 14:45:00 | 704.50 | 699.42 | 694.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-03 11:15:00 | 771.98 | 757.79 | 744.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 742.35 | 760.49 | 760.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 733.85 | 745.02 | 749.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 711.60 | 711.51 | 719.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 703.25 | 711.51 | 719.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 702.00 | 702.96 | 709.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 700.00 | 702.36 | 709.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 665.00 | 692.99 | 694.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 674.90 | 668.29 | 675.45 | SL hit (close>ema200) qty=0.50 sl=668.29 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 686.45 | 679.57 | 678.83 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 671.70 | 677.58 | 678.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 670.10 | 676.09 | 677.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 654.15 | 647.76 | 656.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 654.15 | 647.76 | 656.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 662.85 | 650.78 | 657.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 662.85 | 650.78 | 657.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 667.15 | 654.05 | 658.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 667.15 | 654.05 | 658.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 675.35 | 662.29 | 661.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 678.95 | 667.88 | 664.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 681.65 | 683.50 | 676.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 680.50 | 683.50 | 676.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 722.25 | 707.72 | 697.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 722.25 | 707.72 | 697.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 729.65 | 737.35 | 724.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:30:00 | 724.25 | 737.35 | 724.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 714.05 | 730.91 | 723.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 714.05 | 730.91 | 723.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 716.50 | 728.03 | 722.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:30:00 | 710.30 | 728.03 | 722.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 713.90 | 723.28 | 721.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:45:00 | 712.70 | 723.28 | 721.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 725.65 | 723.41 | 721.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 731.30 | 723.23 | 721.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:15:00 | 729.50 | 723.52 | 722.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:45:00 | 727.55 | 725.53 | 723.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 15:00:00 | 726.00 | 726.84 | 724.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 723.90 | 726.25 | 724.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 721.05 | 726.25 | 724.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 717.60 | 724.52 | 723.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 717.60 | 724.52 | 723.95 | SL hit (close<static) qty=1.00 sl=719.60 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 713.00 | 722.22 | 722.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 711.40 | 720.05 | 721.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 678.00 | 655.82 | 663.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 678.00 | 655.82 | 663.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 678.00 | 655.82 | 663.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 678.00 | 655.82 | 663.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 685.25 | 661.71 | 665.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 685.25 | 661.71 | 665.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 659.75 | 666.17 | 667.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:30:00 | 658.15 | 665.03 | 666.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 671.40 | 666.30 | 667.03 | SL hit (close>static) qty=1.00 sl=668.45 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 09:15:00 | 674.05 | 667.03 | 666.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 693.60 | 682.60 | 676.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 700.50 | 708.36 | 701.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 700.50 | 708.36 | 701.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 700.50 | 708.36 | 701.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 700.50 | 708.36 | 701.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 702.60 | 707.21 | 701.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 703.65 | 707.21 | 701.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 697.35 | 705.24 | 701.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 697.35 | 705.24 | 701.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 692.15 | 702.62 | 700.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 692.15 | 702.62 | 700.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 693.05 | 697.91 | 698.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 685.90 | 695.51 | 697.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 684.00 | 679.91 | 686.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 684.00 | 679.91 | 686.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 684.00 | 679.91 | 686.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 684.30 | 679.91 | 686.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 680.00 | 679.78 | 685.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:30:00 | 683.00 | 679.78 | 685.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 677.20 | 679.26 | 684.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:15:00 | 667.95 | 683.36 | 684.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:00:00 | 669.95 | 680.68 | 682.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 634.55 | 671.16 | 678.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 636.45 | 671.16 | 678.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 602.96 | 632.75 | 652.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 611.15 | 606.78 | 606.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 614.40 | 609.16 | 607.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 600.70 | 607.47 | 607.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 600.70 | 607.47 | 607.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 600.70 | 607.47 | 607.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 600.70 | 607.47 | 607.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 600.85 | 606.14 | 606.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 597.85 | 604.48 | 605.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 602.90 | 602.72 | 604.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 602.90 | 602.72 | 604.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 602.90 | 602.72 | 604.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 604.30 | 602.72 | 604.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 604.95 | 603.20 | 604.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 608.15 | 603.20 | 604.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 606.50 | 603.86 | 604.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 608.40 | 603.86 | 604.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 602.85 | 600.10 | 602.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 602.35 | 600.10 | 602.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 606.95 | 601.47 | 602.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 605.40 | 601.47 | 602.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 608.30 | 602.84 | 603.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 608.95 | 602.84 | 603.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 609.65 | 604.20 | 603.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 614.25 | 606.21 | 604.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 634.35 | 634.60 | 626.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 634.35 | 634.60 | 626.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 632.75 | 634.41 | 627.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 636.25 | 634.41 | 627.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:00:00 | 638.25 | 631.06 | 629.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 657.25 | 663.22 | 663.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 657.25 | 663.22 | 663.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 648.20 | 660.21 | 661.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 662.50 | 658.40 | 660.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 662.50 | 658.40 | 660.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 662.50 | 658.40 | 660.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 662.50 | 658.40 | 660.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 661.55 | 659.03 | 660.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 661.75 | 659.03 | 660.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 662.35 | 659.69 | 660.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 670.75 | 659.69 | 660.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 675.25 | 662.81 | 662.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 682.85 | 670.99 | 666.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 672.45 | 683.90 | 678.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 672.45 | 683.90 | 678.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 672.45 | 683.90 | 678.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 672.45 | 683.90 | 678.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 681.20 | 683.36 | 678.30 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 655.30 | 675.30 | 676.30 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 684.50 | 672.82 | 672.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 690.65 | 681.89 | 678.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 15:15:00 | 682.05 | 682.92 | 679.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 09:15:00 | 681.20 | 682.92 | 679.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 675.00 | 681.34 | 679.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 675.20 | 681.34 | 679.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 683.30 | 681.73 | 679.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 11:30:00 | 686.75 | 682.91 | 680.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 692.75 | 698.42 | 698.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 692.75 | 698.42 | 698.75 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 708.25 | 700.08 | 699.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 709.60 | 701.99 | 700.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 705.65 | 707.30 | 704.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:15:00 | 704.45 | 707.30 | 704.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 708.30 | 707.50 | 704.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:45:00 | 709.55 | 708.58 | 705.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 695.95 | 706.53 | 705.59 | SL hit (close<static) qty=1.00 sl=702.45 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 698.05 | 704.84 | 704.90 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 711.40 | 702.99 | 702.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 720.75 | 709.64 | 706.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 713.85 | 714.49 | 710.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:00:00 | 713.85 | 714.49 | 710.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 711.20 | 714.56 | 711.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:00:00 | 711.20 | 714.56 | 711.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 707.20 | 713.09 | 711.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:30:00 | 707.45 | 713.09 | 711.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 703.70 | 711.21 | 710.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 703.70 | 711.21 | 710.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 13:15:00 | 703.80 | 709.73 | 709.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 14:15:00 | 699.80 | 707.74 | 708.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 719.70 | 708.74 | 709.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 719.70 | 708.74 | 709.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 719.70 | 708.74 | 709.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 719.70 | 708.74 | 709.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 718.50 | 710.69 | 709.96 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 707.00 | 711.82 | 711.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 705.60 | 710.57 | 711.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 707.75 | 707.02 | 708.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 13:15:00 | 707.75 | 707.02 | 708.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 707.75 | 707.02 | 708.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 707.75 | 707.02 | 708.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 706.75 | 706.96 | 708.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 701.55 | 706.61 | 708.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 709.35 | 707.42 | 708.46 | SL hit (close>static) qty=1.00 sl=708.75 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 694.90 | 687.71 | 687.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 13:15:00 | 704.60 | 695.88 | 691.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 690.70 | 699.70 | 696.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 13:15:00 | 690.70 | 699.70 | 696.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 690.70 | 699.70 | 696.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 690.70 | 699.70 | 696.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 693.25 | 698.41 | 696.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 691.70 | 698.41 | 696.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 679.30 | 693.70 | 694.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 673.15 | 682.46 | 687.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 676.30 | 675.88 | 680.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 10:00:00 | 676.30 | 675.88 | 680.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 683.95 | 677.50 | 681.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 683.95 | 677.50 | 681.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 685.05 | 679.01 | 681.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:45:00 | 685.45 | 679.01 | 681.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 701.30 | 683.46 | 683.25 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 675.60 | 684.54 | 685.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 674.25 | 680.46 | 682.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 674.60 | 674.41 | 677.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 674.60 | 674.41 | 677.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 676.85 | 674.89 | 677.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 672.35 | 674.89 | 677.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 12:15:00 | 669.05 | 665.05 | 664.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 669.05 | 665.05 | 664.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 670.80 | 666.47 | 665.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 659.15 | 665.85 | 665.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 659.15 | 665.85 | 665.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 659.15 | 665.85 | 665.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 659.15 | 665.85 | 665.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 658.40 | 664.36 | 664.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 654.70 | 661.45 | 663.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 657.90 | 657.25 | 659.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 657.90 | 657.25 | 659.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 657.90 | 657.25 | 659.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 659.80 | 657.25 | 659.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 660.20 | 657.84 | 659.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 660.20 | 657.84 | 659.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 660.95 | 658.46 | 659.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 662.00 | 658.46 | 659.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 658.95 | 658.56 | 659.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 656.90 | 658.64 | 659.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:00:00 | 656.70 | 657.26 | 658.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 666.60 | 658.76 | 658.94 | SL hit (close>static) qty=1.00 sl=665.80 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 663.75 | 659.76 | 659.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 674.20 | 662.65 | 660.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 11:15:00 | 693.65 | 694.05 | 688.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 12:00:00 | 693.65 | 694.05 | 688.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 692.40 | 695.55 | 692.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 692.40 | 695.55 | 692.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 694.55 | 695.35 | 692.77 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 687.60 | 692.25 | 692.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 684.80 | 690.76 | 691.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 676.25 | 674.05 | 680.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:00:00 | 676.25 | 674.05 | 680.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 681.40 | 675.52 | 680.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:30:00 | 687.35 | 675.52 | 680.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 683.40 | 677.10 | 680.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:45:00 | 685.55 | 677.10 | 680.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 681.25 | 677.98 | 680.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:30:00 | 682.90 | 677.98 | 680.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 681.95 | 678.78 | 680.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 681.95 | 678.78 | 680.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 682.00 | 679.42 | 680.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 678.65 | 679.42 | 680.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 683.30 | 680.20 | 680.93 | SL hit (close>static) qty=1.00 sl=682.35 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 693.00 | 683.61 | 682.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 694.70 | 686.84 | 684.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 683.55 | 687.78 | 685.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 683.55 | 687.78 | 685.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 683.55 | 687.78 | 685.39 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 677.75 | 683.11 | 683.80 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 693.45 | 683.76 | 683.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 695.45 | 687.55 | 685.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 15:15:00 | 689.50 | 691.27 | 688.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 15:15:00 | 689.50 | 691.27 | 688.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 689.50 | 691.27 | 688.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 686.20 | 691.27 | 688.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 683.25 | 689.67 | 688.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 682.65 | 689.67 | 688.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 687.05 | 689.14 | 688.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 692.80 | 689.88 | 688.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 11:15:00 | 702.50 | 706.58 | 706.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 11:15:00 | 702.50 | 706.58 | 706.87 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 714.55 | 706.37 | 706.37 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 706.70 | 707.29 | 707.32 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 13:15:00 | 708.55 | 707.54 | 707.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 709.35 | 707.90 | 707.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 707.70 | 707.86 | 707.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 15:15:00 | 707.70 | 707.86 | 707.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 707.70 | 707.86 | 707.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 708.30 | 707.86 | 707.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 709.10 | 708.11 | 707.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 711.50 | 708.16 | 707.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 704.95 | 707.57 | 707.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 704.95 | 707.57 | 707.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 703.95 | 706.04 | 706.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 710.05 | 706.01 | 706.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 710.05 | 706.01 | 706.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 710.05 | 706.01 | 706.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 710.05 | 706.01 | 706.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 709.40 | 706.69 | 706.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 696.70 | 706.69 | 706.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 685.55 | 687.57 | 691.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:45:00 | 685.05 | 686.52 | 690.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 685.25 | 686.14 | 689.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:00:00 | 685.30 | 685.97 | 689.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:30:00 | 685.20 | 683.79 | 686.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 686.60 | 684.35 | 686.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 686.60 | 684.35 | 686.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 688.75 | 685.23 | 686.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 688.75 | 685.23 | 686.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 685.20 | 685.22 | 686.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 684.55 | 685.22 | 686.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 10:15:00 | 683.50 | 680.43 | 681.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 686.90 | 682.57 | 682.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 686.90 | 682.57 | 682.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 691.55 | 685.50 | 683.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 691.55 | 691.56 | 689.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:15:00 | 690.00 | 691.56 | 689.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 688.20 | 690.89 | 688.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 688.20 | 690.89 | 688.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 690.05 | 690.72 | 689.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 691.30 | 690.51 | 689.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 683.25 | 688.77 | 688.54 | SL hit (close<static) qty=1.00 sl=687.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 685.30 | 688.08 | 688.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 679.95 | 684.70 | 686.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 661.10 | 660.43 | 666.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 661.05 | 660.43 | 666.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 661.95 | 660.93 | 665.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 664.30 | 660.93 | 665.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 656.25 | 649.44 | 651.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 656.65 | 649.44 | 651.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 653.15 | 650.19 | 651.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 652.00 | 650.19 | 651.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 654.30 | 651.01 | 651.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 650.00 | 651.01 | 651.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 655.10 | 651.89 | 652.00 | SL hit (close>static) qty=1.00 sl=654.50 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 659.70 | 653.45 | 652.70 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 647.50 | 653.83 | 654.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 644.75 | 650.98 | 652.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 647.25 | 646.30 | 649.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 653.50 | 647.80 | 649.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 653.50 | 647.80 | 649.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 653.50 | 647.80 | 649.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 654.30 | 649.10 | 649.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 654.45 | 649.10 | 649.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 649.00 | 649.23 | 649.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 649.25 | 649.23 | 649.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 649.00 | 649.18 | 649.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 644.90 | 649.18 | 649.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 643.70 | 648.08 | 649.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:00:00 | 639.25 | 645.19 | 647.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 635.20 | 643.01 | 645.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 633.60 | 631.06 | 630.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 633.60 | 631.06 | 630.97 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 12:15:00 | 630.30 | 630.91 | 630.91 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 633.70 | 631.47 | 631.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 637.25 | 632.76 | 631.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 633.50 | 633.58 | 632.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 633.50 | 633.58 | 632.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 633.50 | 633.58 | 632.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 633.50 | 633.58 | 632.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 634.00 | 633.67 | 632.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 633.85 | 633.67 | 632.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 636.95 | 637.30 | 635.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 636.80 | 637.30 | 635.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 635.55 | 636.95 | 635.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 635.55 | 636.95 | 635.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 631.00 | 635.76 | 634.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 631.00 | 635.76 | 634.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 632.35 | 635.08 | 634.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 629.70 | 635.08 | 634.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 634.55 | 638.34 | 637.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 634.55 | 638.34 | 637.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 636.25 | 637.92 | 637.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 631.00 | 637.92 | 637.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 627.35 | 635.81 | 636.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 623.60 | 627.69 | 630.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 624.60 | 623.83 | 627.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 624.60 | 623.83 | 627.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 637.85 | 626.64 | 628.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 637.85 | 626.64 | 628.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 649.00 | 631.11 | 629.97 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 634.95 | 637.93 | 638.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 628.85 | 635.72 | 637.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 632.00 | 631.48 | 634.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 632.00 | 631.48 | 634.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 630.70 | 630.73 | 632.28 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 637.65 | 633.78 | 633.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 654.55 | 639.34 | 636.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 659.80 | 659.86 | 655.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 14:00:00 | 659.80 | 659.86 | 655.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 657.55 | 661.45 | 659.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 657.55 | 661.45 | 659.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 654.50 | 660.06 | 658.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 653.45 | 660.06 | 658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 656.40 | 658.08 | 658.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 652.70 | 657.00 | 657.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 635.00 | 633.04 | 637.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:00:00 | 635.00 | 633.04 | 637.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 629.75 | 632.67 | 635.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 626.00 | 631.48 | 634.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 624.45 | 628.62 | 632.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 624.55 | 628.37 | 631.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 626.00 | 625.59 | 627.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 626.60 | 625.80 | 627.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 624.15 | 626.64 | 627.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 622.85 | 625.87 | 626.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 623.85 | 619.96 | 622.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 629.85 | 621.94 | 622.96 | SL hit (close>static) qty=1.00 sl=628.90 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 631.35 | 623.82 | 623.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 632.50 | 629.00 | 626.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 628.90 | 629.85 | 627.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 628.90 | 629.85 | 627.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 627.30 | 629.31 | 627.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 627.30 | 629.31 | 627.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 622.00 | 627.85 | 627.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 622.00 | 627.85 | 627.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 620.00 | 626.28 | 626.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 616.60 | 624.34 | 625.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 617.05 | 615.13 | 619.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 617.05 | 615.13 | 619.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 614.90 | 611.68 | 614.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 612.00 | 611.75 | 614.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 617.90 | 613.61 | 614.66 | SL hit (close>static) qty=1.00 sl=615.40 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 618.50 | 615.32 | 615.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 626.75 | 621.78 | 619.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 621.85 | 625.42 | 622.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 621.85 | 625.42 | 622.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 621.85 | 625.42 | 622.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 621.85 | 625.42 | 622.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 622.90 | 624.91 | 622.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 624.50 | 623.36 | 622.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 624.45 | 623.63 | 622.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 615.90 | 621.89 | 622.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 615.90 | 621.89 | 622.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 613.55 | 620.22 | 621.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 602.05 | 601.01 | 604.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:45:00 | 602.80 | 601.01 | 604.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 598.25 | 600.76 | 603.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 595.35 | 599.68 | 602.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 594.80 | 597.21 | 600.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 595.15 | 590.14 | 590.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 593.40 | 590.79 | 590.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 593.40 | 590.79 | 590.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 598.95 | 592.42 | 591.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 590.40 | 592.74 | 591.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 590.40 | 592.74 | 591.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 590.40 | 592.74 | 591.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 590.40 | 592.74 | 591.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 594.25 | 593.04 | 591.97 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 590.80 | 591.92 | 591.96 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 596.00 | 592.74 | 592.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 596.65 | 593.52 | 592.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 597.05 | 597.15 | 595.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:15:00 | 594.35 | 597.15 | 595.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 597.35 | 597.19 | 595.32 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 590.40 | 594.31 | 594.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 588.30 | 593.11 | 593.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 593.85 | 592.17 | 593.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 593.85 | 592.17 | 593.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 593.85 | 592.17 | 593.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 593.95 | 592.17 | 593.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 593.60 | 592.46 | 593.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 594.35 | 592.46 | 593.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 593.70 | 593.00 | 593.24 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 595.50 | 593.50 | 593.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 596.50 | 594.48 | 593.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 593.15 | 594.34 | 593.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 593.15 | 594.34 | 593.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 593.15 | 594.34 | 593.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 593.15 | 594.34 | 593.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 593.40 | 594.15 | 593.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 593.40 | 594.15 | 593.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 597.45 | 594.81 | 594.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 600.75 | 596.05 | 594.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 597.10 | 604.91 | 605.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 597.10 | 604.91 | 605.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 594.50 | 600.75 | 603.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 600.30 | 599.83 | 602.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:30:00 | 600.15 | 599.83 | 602.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 596.50 | 599.32 | 601.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:45:00 | 596.20 | 598.68 | 600.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:15:00 | 566.39 | 578.65 | 586.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 576.50 | 576.37 | 582.42 | SL hit (close>ema200) qty=0.50 sl=576.37 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 589.35 | 585.03 | 584.60 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 576.70 | 583.10 | 583.90 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 584.65 | 583.61 | 583.53 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 579.40 | 582.77 | 583.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 577.15 | 580.27 | 581.72 | Break + close below crossover candle low |

### Cycle 113 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 620.45 | 586.78 | 584.17 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 597.40 | 599.95 | 600.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 594.25 | 598.81 | 599.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 593.60 | 593.49 | 595.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:30:00 | 593.10 | 593.49 | 595.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 589.55 | 591.91 | 594.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:30:00 | 588.10 | 591.53 | 594.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 588.00 | 590.87 | 593.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 587.75 | 590.39 | 592.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 586.90 | 589.17 | 590.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 595.70 | 590.29 | 591.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 595.70 | 590.29 | 591.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 595.00 | 591.23 | 591.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 593.45 | 591.23 | 591.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 594.30 | 591.85 | 591.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 594.30 | 591.85 | 591.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 596.90 | 593.58 | 592.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 606.00 | 606.65 | 603.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 607.30 | 606.65 | 603.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 608.05 | 606.93 | 603.49 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 600.50 | 602.20 | 602.27 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 604.30 | 602.63 | 602.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 606.55 | 603.48 | 602.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 602.95 | 605.40 | 604.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 602.95 | 605.40 | 604.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 602.95 | 605.40 | 604.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 602.95 | 605.40 | 604.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 602.00 | 604.72 | 603.98 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 601.45 | 603.35 | 603.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 600.00 | 602.68 | 603.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 603.60 | 602.87 | 603.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 603.60 | 602.87 | 603.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 603.60 | 602.87 | 603.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 603.60 | 602.87 | 603.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 602.00 | 602.69 | 603.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 601.25 | 602.69 | 603.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 591.65 | 587.37 | 586.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 591.65 | 587.37 | 586.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 595.00 | 588.89 | 587.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 12:15:00 | 592.15 | 597.27 | 593.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 592.15 | 597.27 | 593.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 592.15 | 597.27 | 593.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 592.15 | 597.27 | 593.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 588.10 | 595.44 | 592.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 588.55 | 595.44 | 592.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 584.75 | 593.30 | 592.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 584.75 | 593.30 | 592.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 579.00 | 590.44 | 591.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 575.55 | 581.80 | 585.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 555.65 | 555.29 | 562.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:30:00 | 558.10 | 555.29 | 562.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 557.55 | 557.49 | 560.55 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 565.00 | 562.08 | 561.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 567.60 | 563.19 | 562.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 566.50 | 567.18 | 565.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:45:00 | 566.50 | 567.18 | 565.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 566.65 | 567.07 | 565.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 565.00 | 567.07 | 565.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 566.00 | 566.85 | 565.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 563.50 | 566.85 | 565.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 563.40 | 566.16 | 565.64 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 564.10 | 565.30 | 565.31 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 565.60 | 565.36 | 565.33 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 564.30 | 565.15 | 565.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 563.40 | 564.80 | 565.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 564.60 | 564.57 | 564.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 564.60 | 564.57 | 564.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 564.60 | 564.57 | 564.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 564.60 | 564.57 | 564.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 565.00 | 564.65 | 564.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 565.40 | 564.65 | 564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 561.45 | 564.01 | 564.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 558.20 | 562.85 | 564.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 558.70 | 562.34 | 563.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 570.35 | 564.06 | 563.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 570.35 | 564.06 | 563.52 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 558.80 | 562.66 | 563.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 15:15:00 | 557.80 | 561.05 | 562.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 560.05 | 559.77 | 561.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:45:00 | 560.05 | 559.77 | 561.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 558.65 | 559.55 | 560.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 561.00 | 559.55 | 560.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 555.45 | 555.02 | 556.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 550.05 | 555.30 | 556.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 11:15:00 | 522.55 | 526.60 | 532.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 522.40 | 520.95 | 524.83 | SL hit (close>ema200) qty=0.50 sl=520.95 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 528.00 | 525.66 | 525.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 528.85 | 526.65 | 525.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 526.65 | 527.31 | 526.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 526.65 | 527.31 | 526.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 526.65 | 527.31 | 526.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 525.00 | 527.31 | 526.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 528.75 | 527.60 | 526.68 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 522.50 | 526.66 | 526.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 517.75 | 524.88 | 526.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 509.40 | 507.80 | 512.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 509.65 | 507.80 | 512.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 509.85 | 507.90 | 511.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 510.60 | 507.90 | 511.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 488.00 | 488.65 | 493.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 486.90 | 488.65 | 493.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 495.00 | 491.23 | 492.74 | SL hit (close>static) qty=1.00 sl=493.90 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 496.75 | 491.80 | 491.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 15:15:00 | 498.25 | 494.32 | 492.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 498.60 | 498.72 | 495.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 498.60 | 498.72 | 495.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 498.60 | 498.72 | 495.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 501.65 | 498.72 | 495.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 496.80 | 498.34 | 495.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 497.05 | 498.34 | 495.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 491.35 | 496.94 | 495.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 491.35 | 496.94 | 495.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 494.00 | 496.35 | 495.32 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 490.30 | 494.65 | 494.70 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 505.95 | 496.28 | 495.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 09:15:00 | 521.70 | 514.89 | 508.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 548.40 | 551.88 | 547.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 548.40 | 551.88 | 547.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 548.40 | 551.88 | 547.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 548.40 | 551.88 | 547.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 548.30 | 551.17 | 547.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 544.55 | 551.17 | 547.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 543.95 | 549.72 | 546.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 540.10 | 549.72 | 546.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 544.10 | 548.60 | 546.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:15:00 | 541.65 | 548.60 | 546.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 545.00 | 547.13 | 546.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 540.85 | 547.13 | 546.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 538.25 | 545.36 | 545.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 526.45 | 537.65 | 541.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 542.60 | 538.18 | 540.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 542.60 | 538.18 | 540.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 542.60 | 538.18 | 540.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 542.60 | 538.18 | 540.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 539.05 | 538.35 | 540.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:30:00 | 535.65 | 537.13 | 539.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 539.55 | 530.19 | 529.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 539.55 | 530.19 | 529.05 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 520.90 | 529.97 | 530.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 14:15:00 | 519.05 | 525.13 | 528.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 518.95 | 518.11 | 522.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 518.95 | 518.11 | 522.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 521.90 | 518.69 | 521.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 520.80 | 518.69 | 521.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 520.70 | 519.09 | 521.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:30:00 | 521.40 | 519.09 | 521.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 514.10 | 518.09 | 520.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 511.90 | 516.92 | 520.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 512.75 | 516.22 | 518.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 503.55 | 514.08 | 515.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 486.30 | 491.80 | 496.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 487.11 | 491.80 | 496.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 478.37 | 491.80 | 496.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 487.75 | 487.01 | 491.83 | SL hit (close>ema200) qty=0.50 sl=487.01 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 469.45 | 463.48 | 463.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 471.15 | 466.08 | 464.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 461.30 | 471.71 | 469.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 461.30 | 471.71 | 469.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 461.30 | 471.71 | 469.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 461.30 | 471.71 | 469.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 459.25 | 469.22 | 468.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 459.25 | 469.22 | 468.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 459.70 | 467.31 | 467.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 458.25 | 465.50 | 466.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 460.55 | 460.26 | 463.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:30:00 | 459.40 | 460.26 | 463.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 445.20 | 443.34 | 449.51 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 466.00 | 453.47 | 451.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 468.80 | 456.54 | 453.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 456.65 | 460.20 | 456.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 456.65 | 460.20 | 456.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 456.65 | 460.20 | 456.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 456.65 | 460.20 | 456.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 454.80 | 459.12 | 456.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 453.50 | 459.12 | 456.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 457.75 | 458.85 | 456.76 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 437.55 | 452.59 | 454.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 433.20 | 443.76 | 449.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 445.10 | 441.73 | 446.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 445.10 | 441.73 | 446.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 445.10 | 441.73 | 446.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 443.55 | 441.73 | 446.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 443.45 | 442.56 | 445.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 442.00 | 440.65 | 442.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 453.20 | 443.62 | 443.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 453.20 | 443.62 | 443.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 458.00 | 446.50 | 444.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 426.55 | 449.53 | 447.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 426.55 | 449.53 | 447.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 426.55 | 449.53 | 447.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 426.90 | 449.53 | 447.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 418.00 | 443.22 | 445.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 11:15:00 | 415.55 | 437.69 | 442.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 440.85 | 426.73 | 433.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 440.85 | 426.73 | 433.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 440.85 | 426.73 | 433.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 441.00 | 426.73 | 433.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 437.90 | 428.97 | 434.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:15:00 | 435.65 | 428.97 | 434.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:45:00 | 434.65 | 430.34 | 434.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 436.20 | 432.61 | 434.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 442.40 | 433.33 | 433.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 11:15:00 | 442.40 | 433.33 | 433.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 12:15:00 | 445.25 | 435.72 | 434.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 437.70 | 439.78 | 437.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 437.70 | 439.78 | 437.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 437.70 | 439.78 | 437.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 445.30 | 439.08 | 437.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 489.83 | 477.11 | 469.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 480.40 | 485.90 | 486.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 478.50 | 481.11 | 483.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 482.20 | 481.33 | 483.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 482.20 | 481.33 | 483.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 486.40 | 482.35 | 483.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 486.40 | 482.35 | 483.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 483.40 | 482.56 | 483.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 479.00 | 483.80 | 483.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 477.35 | 478.08 | 480.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 480.20 | 478.65 | 480.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 480.40 | 479.04 | 480.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 476.25 | 478.48 | 479.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:30:00 | 477.80 | 478.48 | 479.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 471.40 | 467.06 | 470.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 471.40 | 467.06 | 470.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 471.00 | 467.85 | 470.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 470.05 | 467.85 | 470.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 469.00 | 468.13 | 469.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 468.65 | 468.13 | 469.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 469.15 | 468.33 | 469.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 468.60 | 468.33 | 469.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 473.30 | 469.33 | 470.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 473.30 | 469.33 | 470.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 474.85 | 470.43 | 470.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 474.40 | 470.43 | 470.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-07 14:15:00 | 477.50 | 471.84 | 471.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 477.50 | 471.84 | 471.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 479.10 | 473.30 | 471.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 474.25 | 475.08 | 473.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 474.25 | 475.08 | 473.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 474.40 | 474.95 | 473.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 474.40 | 474.95 | 473.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 473.85 | 474.73 | 473.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 473.85 | 474.73 | 473.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 473.00 | 474.38 | 473.45 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-30 09:15:00 | 517.95 | 2024-05-30 14:15:00 | 503.75 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-05-31 09:15:00 | 507.80 | 2024-05-31 11:15:00 | 495.05 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-06-03 10:45:00 | 495.40 | 2024-06-04 12:15:00 | 470.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:30:00 | 495.50 | 2024-06-04 12:15:00 | 470.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 495.40 | 2024-06-04 14:15:00 | 498.20 | STOP_HIT | 0.50 | -0.57% |
| SELL | retest2 | 2024-06-03 11:30:00 | 495.50 | 2024-06-04 14:15:00 | 498.20 | STOP_HIT | 0.50 | -0.54% |
| SELL | retest2 | 2024-06-04 09:15:00 | 485.25 | 2024-06-05 09:15:00 | 520.00 | STOP_HIT | 1.00 | -7.16% |
| SELL | retest2 | 2024-06-04 10:45:00 | 492.35 | 2024-06-05 09:15:00 | 520.00 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2024-06-07 14:30:00 | 519.40 | 2024-06-20 15:15:00 | 538.00 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2024-06-10 09:15:00 | 519.55 | 2024-06-20 15:15:00 | 538.00 | STOP_HIT | 1.00 | 3.55% |
| BUY | retest2 | 2024-06-10 12:15:00 | 519.20 | 2024-06-20 15:15:00 | 538.00 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2024-06-10 12:45:00 | 519.25 | 2024-06-20 15:15:00 | 538.00 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2024-06-11 09:15:00 | 524.50 | 2024-06-20 15:15:00 | 538.00 | STOP_HIT | 1.00 | 2.57% |
| BUY | retest2 | 2024-06-24 12:15:00 | 556.35 | 2024-06-27 13:15:00 | 552.70 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-06-26 12:15:00 | 556.50 | 2024-06-27 13:15:00 | 552.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-06-26 13:45:00 | 556.50 | 2024-06-27 13:15:00 | 552.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-06-27 09:30:00 | 557.25 | 2024-06-27 13:15:00 | 552.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-02 14:30:00 | 576.20 | 2024-07-04 15:15:00 | 565.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-07-11 13:15:00 | 580.10 | 2024-07-18 11:15:00 | 570.95 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-07-12 10:45:00 | 579.45 | 2024-07-18 11:15:00 | 570.95 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-07-15 14:30:00 | 579.30 | 2024-07-18 11:15:00 | 570.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-07-16 09:30:00 | 578.55 | 2024-07-18 11:15:00 | 570.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-07-23 09:30:00 | 564.80 | 2024-07-24 14:15:00 | 582.50 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-07-23 11:45:00 | 566.55 | 2024-07-24 14:15:00 | 582.50 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-07-31 09:15:00 | 592.80 | 2024-08-05 11:15:00 | 588.55 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-07 09:15:00 | 603.55 | 2024-08-08 14:15:00 | 596.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-08 13:30:00 | 600.15 | 2024-08-08 14:15:00 | 596.45 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-09 11:00:00 | 599.05 | 2024-08-12 09:15:00 | 615.40 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-08-09 13:00:00 | 599.15 | 2024-08-12 09:15:00 | 615.40 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-08-09 13:30:00 | 598.40 | 2024-08-12 09:15:00 | 615.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-08-09 14:30:00 | 599.05 | 2024-08-12 09:15:00 | 615.40 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-08-16 09:15:00 | 645.45 | 2024-08-19 10:15:00 | 630.55 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-08-16 09:45:00 | 641.25 | 2024-08-19 10:15:00 | 630.55 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-08-16 11:15:00 | 640.40 | 2024-08-19 10:15:00 | 630.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-08-26 10:30:00 | 657.35 | 2024-08-29 09:15:00 | 652.65 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-26 11:30:00 | 656.35 | 2024-08-29 09:15:00 | 652.65 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-08-26 12:30:00 | 656.85 | 2024-08-29 09:15:00 | 652.65 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-28 13:30:00 | 658.45 | 2024-08-29 09:15:00 | 652.65 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-08-30 12:15:00 | 656.10 | 2024-09-03 15:15:00 | 653.80 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-09-09 09:15:00 | 637.55 | 2024-09-10 09:15:00 | 654.80 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-09-09 10:30:00 | 643.15 | 2024-09-10 09:15:00 | 654.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-09-12 12:15:00 | 667.80 | 2024-09-17 09:15:00 | 663.40 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-09-17 09:15:00 | 668.05 | 2024-09-17 09:15:00 | 663.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2024-09-18 09:30:00 | 680.10 | 2024-09-18 12:15:00 | 672.45 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2024-09-18 10:30:00 | 678.05 | 2024-09-18 12:15:00 | 672.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-19 09:15:00 | 682.90 | 2024-09-26 09:15:00 | 696.75 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2024-09-19 12:00:00 | 680.50 | 2024-09-26 09:15:00 | 696.75 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2024-10-01 10:15:00 | 659.10 | 2024-10-04 09:15:00 | 626.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:30:00 | 659.80 | 2024-10-04 09:15:00 | 626.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 659.10 | 2024-10-09 09:15:00 | 624.65 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2024-10-01 12:30:00 | 659.80 | 2024-10-09 09:15:00 | 624.65 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2024-10-15 13:15:00 | 622.80 | 2024-10-15 14:15:00 | 626.20 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-17 14:30:00 | 613.75 | 2024-10-18 11:15:00 | 624.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-10-18 10:45:00 | 616.65 | 2024-10-18 11:15:00 | 624.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest1 | 2024-10-25 10:00:00 | 599.35 | 2024-10-29 12:15:00 | 569.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-25 12:45:00 | 594.00 | 2024-10-29 13:15:00 | 564.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-25 10:00:00 | 599.35 | 2024-10-29 15:15:00 | 578.20 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest1 | 2024-10-25 12:45:00 | 594.00 | 2024-10-29 15:15:00 | 578.20 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-10-30 13:45:00 | 580.30 | 2024-11-05 11:15:00 | 583.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-11-12 09:15:00 | 647.45 | 2024-11-14 12:15:00 | 613.20 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest2 | 2024-11-29 10:45:00 | 648.00 | 2024-12-11 12:15:00 | 712.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 13:30:00 | 646.50 | 2024-12-11 12:15:00 | 711.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 10:30:00 | 648.75 | 2024-12-11 12:15:00 | 713.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-13 14:45:00 | 680.05 | 2024-12-16 12:15:00 | 690.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-12-19 13:45:00 | 678.20 | 2024-12-23 10:15:00 | 689.05 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-19 15:15:00 | 676.00 | 2024-12-23 10:15:00 | 689.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-12-20 09:30:00 | 677.80 | 2024-12-23 10:15:00 | 689.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-20 12:00:00 | 678.50 | 2024-12-23 10:15:00 | 689.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-26 14:00:00 | 701.80 | 2025-01-03 11:15:00 | 771.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-26 14:45:00 | 704.50 | 2025-01-06 09:15:00 | 774.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-16 10:45:00 | 700.00 | 2025-01-21 09:15:00 | 665.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 10:45:00 | 700.00 | 2025-01-23 09:15:00 | 674.90 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2025-02-05 09:15:00 | 731.30 | 2025-02-06 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-02-05 10:15:00 | 729.50 | 2025-02-06 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-02-05 11:45:00 | 727.55 | 2025-02-06 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-02-05 15:00:00 | 726.00 | 2025-02-06 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-02-14 09:30:00 | 658.15 | 2025-02-14 10:15:00 | 671.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-02-14 11:30:00 | 659.05 | 2025-02-14 15:15:00 | 670.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-14 13:45:00 | 658.95 | 2025-02-14 15:15:00 | 670.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-02-27 14:15:00 | 667.95 | 2025-02-28 09:15:00 | 634.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 15:00:00 | 669.95 | 2025-02-28 09:15:00 | 636.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 14:15:00 | 667.95 | 2025-03-03 09:15:00 | 602.96 | TARGET_HIT | 0.50 | 9.73% |
| SELL | retest2 | 2025-02-27 15:00:00 | 669.95 | 2025-03-03 10:15:00 | 601.16 | TARGET_HIT | 0.50 | 10.27% |
| BUY | retest2 | 2025-03-20 10:15:00 | 636.25 | 2025-04-01 09:15:00 | 657.25 | STOP_HIT | 1.00 | 3.30% |
| BUY | retest2 | 2025-03-24 10:00:00 | 638.25 | 2025-04-01 09:15:00 | 657.25 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2025-04-11 11:30:00 | 686.75 | 2025-04-23 10:15:00 | 692.75 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-04-24 13:45:00 | 709.55 | 2025-04-25 10:15:00 | 695.95 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-08 09:15:00 | 701.55 | 2025-05-08 10:15:00 | 709.35 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-05-08 11:45:00 | 702.95 | 2025-05-08 15:15:00 | 667.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:45:00 | 702.95 | 2025-05-09 15:15:00 | 673.70 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-05-23 09:15:00 | 672.35 | 2025-05-29 12:15:00 | 669.05 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-06-03 11:15:00 | 656.90 | 2025-06-04 09:15:00 | 666.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-06-03 15:00:00 | 656.70 | 2025-06-04 09:15:00 | 666.60 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-06-16 09:15:00 | 678.65 | 2025-06-16 09:15:00 | 683.30 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-23 12:00:00 | 692.80 | 2025-06-30 11:15:00 | 702.50 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-03 11:15:00 | 711.50 | 2025-07-03 15:15:00 | 704.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-09 10:45:00 | 685.05 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-07-09 12:30:00 | 685.25 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-09 14:00:00 | 685.30 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-10 12:30:00 | 685.20 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-11 09:15:00 | 684.55 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-15 10:15:00 | 683.50 | 2025-07-15 11:15:00 | 686.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-17 14:30:00 | 691.30 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-30 09:15:00 | 650.00 | 2025-07-30 11:15:00 | 655.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-08-05 13:00:00 | 639.25 | 2025-08-12 11:15:00 | 633.60 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-08-06 09:15:00 | 635.20 | 2025-08-12 11:15:00 | 633.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-17 11:15:00 | 626.00 | 2025-09-24 10:15:00 | 629.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-17 14:00:00 | 624.45 | 2025-09-24 10:15:00 | 629.85 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-18 09:15:00 | 624.55 | 2025-09-24 10:15:00 | 629.85 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-19 10:30:00 | 626.00 | 2025-09-24 11:15:00 | 631.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-22 13:45:00 | 624.15 | 2025-09-24 11:15:00 | 631.35 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-22 14:30:00 | 622.85 | 2025-09-24 11:15:00 | 631.35 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-24 10:00:00 | 623.85 | 2025-09-24 11:15:00 | 631.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-30 12:00:00 | 612.00 | 2025-09-30 14:15:00 | 617.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-06 15:15:00 | 624.50 | 2025-10-07 12:15:00 | 615.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-10-07 10:30:00 | 624.45 | 2025-10-07 12:15:00 | 615.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-13 11:00:00 | 595.35 | 2025-10-16 15:15:00 | 593.40 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-10-14 09:45:00 | 594.80 | 2025-10-16 15:15:00 | 593.40 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-10-16 15:00:00 | 595.15 | 2025-10-16 15:15:00 | 593.40 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-28 14:45:00 | 600.75 | 2025-10-31 14:15:00 | 597.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-04 10:45:00 | 596.20 | 2025-11-07 11:15:00 | 566.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:45:00 | 596.20 | 2025-11-07 15:15:00 | 576.50 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-11-21 10:30:00 | 588.10 | 2025-11-25 11:15:00 | 594.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-21 13:30:00 | 588.00 | 2025-11-25 11:15:00 | 594.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-24 10:15:00 | 587.75 | 2025-11-25 11:15:00 | 594.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-11-24 14:45:00 | 586.90 | 2025-11-25 11:15:00 | 594.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-25 11:15:00 | 593.45 | 2025-11-25 11:15:00 | 594.30 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-12-03 09:15:00 | 601.25 | 2025-12-11 11:15:00 | 591.65 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-12-26 13:00:00 | 558.20 | 2025-12-30 09:15:00 | 570.35 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-12-26 14:15:00 | 558.70 | 2025-12-30 09:15:00 | 570.35 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-01-02 13:15:00 | 550.05 | 2026-01-09 11:15:00 | 522.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 13:15:00 | 550.05 | 2026-01-12 14:15:00 | 522.40 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2026-01-28 10:15:00 | 486.90 | 2026-01-28 15:15:00 | 495.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-29 10:30:00 | 486.65 | 2026-01-30 11:15:00 | 496.75 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-01-29 11:15:00 | 486.75 | 2026-01-30 11:15:00 | 496.75 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-01-29 12:15:00 | 486.30 | 2026-01-30 11:15:00 | 496.75 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-02-13 14:30:00 | 535.65 | 2026-02-20 13:15:00 | 539.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-02-26 12:45:00 | 511.90 | 2026-03-09 09:15:00 | 486.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 512.75 | 2026-03-09 09:15:00 | 487.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 503.55 | 2026-03-09 09:15:00 | 478.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 511.90 | 2026-03-09 14:15:00 | 487.75 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2026-02-27 09:15:00 | 512.75 | 2026-03-09 14:15:00 | 487.75 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2026-03-02 09:15:00 | 503.55 | 2026-03-09 14:15:00 | 487.75 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-04-01 10:15:00 | 443.55 | 2026-04-06 10:15:00 | 453.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-01 13:15:00 | 443.45 | 2026-04-06 10:15:00 | 453.20 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-02 15:15:00 | 442.00 | 2026-04-06 10:15:00 | 453.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-08 11:15:00 | 435.65 | 2026-04-10 11:15:00 | 442.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-08 11:45:00 | 434.65 | 2026-04-10 11:15:00 | 442.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-08 14:00:00 | 436.20 | 2026-04-10 11:15:00 | 442.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-15 09:15:00 | 445.30 | 2026-04-22 12:15:00 | 489.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 479.00 | 2026-05-07 14:15:00 | 477.50 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2026-04-30 15:00:00 | 477.35 | 2026-05-07 14:15:00 | 477.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-05-04 10:00:00 | 480.20 | 2026-05-07 14:15:00 | 477.50 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2026-05-04 11:30:00 | 480.40 | 2026-05-07 14:15:00 | 477.50 | STOP_HIT | 1.00 | 0.60% |
