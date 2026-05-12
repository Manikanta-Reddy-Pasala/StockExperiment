# REC Ltd. (RECLTD)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 359.30
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
| ALERT2 | 101 |
| ALERT2_SKIP | 45 |
| ALERT3 | 271 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 111 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 91
- **Target hits / Stop hits / Partials:** 6 / 107 / 6
- **Avg / median % per leg:** -0.12% / -0.86%
- **Sum % (uncompounded):** -14.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 8 | 18.2% | 3 | 41 | 0 | -0.27% | -11.7% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.87% | -1.7% |
| BUY @ 3rd Alert (retest2) | 42 | 7 | 16.7% | 3 | 39 | 0 | -0.24% | -10.0% |
| SELL (all) | 75 | 20 | 26.7% | 3 | 66 | 6 | -0.03% | -2.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 75 | 20 | 26.7% | 3 | 66 | 6 | -0.03% | -2.3% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.87% | -1.7% |
| retest2 (combined) | 117 | 27 | 23.1% | 6 | 105 | 6 | -0.11% | -12.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 519.60 | 516.37 | 516.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 522.70 | 518.47 | 517.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 527.50 | 527.87 | 523.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 534.65 | 527.87 | 523.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 553.30 | 545.78 | 540.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 538.15 | 544.38 | 542.77 | SL hit (close<ema400) qty=1.00 sl=542.77 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 535.35 | 541.76 | 541.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 533.20 | 540.04 | 541.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 544.90 | 539.46 | 540.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 544.90 | 539.46 | 540.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 544.90 | 539.46 | 540.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 544.90 | 539.46 | 540.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 538.50 | 539.26 | 540.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:30:00 | 536.55 | 539.13 | 539.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:00:00 | 537.95 | 538.90 | 539.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:15:00 | 537.90 | 538.89 | 539.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:45:00 | 538.00 | 538.62 | 539.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 539.85 | 538.61 | 539.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 541.25 | 538.61 | 539.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 543.00 | 539.48 | 539.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 543.00 | 539.48 | 539.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-24 11:15:00 | 553.70 | 542.33 | 540.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 553.70 | 542.33 | 540.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 558.05 | 545.47 | 542.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 563.30 | 567.81 | 560.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 12:00:00 | 563.30 | 567.81 | 560.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 567.00 | 567.65 | 561.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 562.50 | 567.65 | 561.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 567.25 | 570.14 | 567.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 569.70 | 570.14 | 567.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 562.90 | 568.69 | 567.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 562.90 | 568.69 | 567.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 562.85 | 567.52 | 566.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:15:00 | 561.40 | 567.52 | 566.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 559.95 | 566.01 | 566.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 552.05 | 562.28 | 564.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 542.00 | 541.87 | 550.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 542.00 | 541.87 | 550.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 591.50 | 550.65 | 552.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 591.50 | 550.65 | 552.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 595.25 | 559.57 | 556.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 603.85 | 574.50 | 564.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 562.95 | 584.16 | 573.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 562.95 | 584.16 | 573.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 562.95 | 584.16 | 573.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 549.80 | 584.16 | 573.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 513.85 | 570.10 | 567.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 513.85 | 570.10 | 567.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 467.75 | 549.63 | 558.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 453.50 | 511.40 | 537.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 494.00 | 468.08 | 490.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 494.00 | 468.08 | 490.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 494.00 | 468.08 | 490.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 490.70 | 468.08 | 490.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 505.45 | 475.55 | 492.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 505.45 | 475.55 | 492.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 495.50 | 479.54 | 492.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:30:00 | 490.75 | 480.97 | 492.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 491.30 | 482.70 | 491.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 09:15:00 | 508.15 | 494.04 | 492.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 508.15 | 494.04 | 492.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 12:15:00 | 512.50 | 500.93 | 496.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 11:15:00 | 509.60 | 510.25 | 506.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 11:30:00 | 509.85 | 510.25 | 506.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 525.10 | 528.36 | 523.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 525.10 | 528.36 | 523.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 529.30 | 528.55 | 523.83 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 519.70 | 523.69 | 523.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 515.95 | 520.45 | 522.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 518.55 | 511.45 | 514.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 518.55 | 511.45 | 514.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 518.55 | 511.45 | 514.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 518.55 | 511.45 | 514.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 516.55 | 512.47 | 514.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:15:00 | 513.00 | 513.86 | 515.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:30:00 | 513.05 | 513.31 | 514.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 11:15:00 | 513.20 | 513.12 | 514.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 523.25 | 516.41 | 515.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 523.25 | 516.41 | 515.63 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 516.10 | 517.36 | 517.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 15:15:00 | 513.80 | 516.65 | 517.08 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 525.85 | 518.49 | 517.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 539.05 | 527.97 | 524.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 536.95 | 541.38 | 535.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 536.95 | 541.38 | 535.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 535.75 | 540.26 | 535.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:30:00 | 533.10 | 540.26 | 535.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 536.90 | 539.59 | 535.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 540.45 | 539.44 | 535.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 539.50 | 539.44 | 535.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-08 09:15:00 | 594.50 | 583.72 | 573.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 629.85 | 632.97 | 633.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 625.10 | 630.19 | 631.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 11:15:00 | 620.20 | 618.93 | 624.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 12:00:00 | 620.20 | 618.93 | 624.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 604.95 | 612.67 | 619.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:30:00 | 615.95 | 612.67 | 619.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 616.50 | 606.22 | 611.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 616.50 | 606.22 | 611.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 614.00 | 607.78 | 611.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 617.85 | 607.78 | 611.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 618.35 | 612.50 | 613.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 618.90 | 612.50 | 613.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 619.80 | 614.93 | 614.28 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 592.95 | 610.26 | 612.35 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 611.30 | 607.14 | 607.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 616.70 | 609.05 | 607.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 636.10 | 636.25 | 627.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 636.10 | 636.25 | 627.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 636.70 | 641.73 | 638.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 636.70 | 641.73 | 638.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 634.95 | 640.38 | 638.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 634.95 | 640.38 | 638.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 628.00 | 635.08 | 635.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 625.35 | 633.14 | 634.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 590.90 | 587.99 | 601.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 590.90 | 587.99 | 601.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 590.90 | 587.99 | 601.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 585.10 | 587.44 | 600.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 585.25 | 587.35 | 598.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:45:00 | 584.10 | 578.87 | 589.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 585.35 | 585.29 | 588.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 586.05 | 585.75 | 588.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:15:00 | 589.85 | 585.75 | 588.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 591.40 | 586.88 | 588.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:30:00 | 591.65 | 586.88 | 588.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 587.50 | 587.00 | 588.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 591.40 | 587.00 | 588.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 584.85 | 586.57 | 588.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 585.35 | 586.57 | 588.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 588.75 | 586.20 | 587.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 590.85 | 586.20 | 587.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 587.60 | 586.90 | 587.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 586.40 | 587.55 | 587.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 581.35 | 587.49 | 587.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 579.80 | 572.92 | 572.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 579.80 | 572.92 | 572.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 588.35 | 576.00 | 574.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 594.15 | 595.97 | 590.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:45:00 | 594.40 | 595.97 | 590.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 592.30 | 594.72 | 592.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 592.30 | 594.72 | 592.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 594.60 | 594.70 | 592.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 595.90 | 594.94 | 592.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:00:00 | 595.25 | 594.98 | 593.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 589.80 | 593.56 | 592.93 | SL hit (close<static) qty=1.00 sl=592.15 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 590.40 | 592.39 | 592.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 586.35 | 591.18 | 591.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 591.80 | 590.67 | 591.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 591.80 | 590.67 | 591.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 591.80 | 590.67 | 591.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 589.05 | 590.67 | 591.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 597.25 | 591.15 | 591.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 597.25 | 591.15 | 591.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 602.30 | 594.36 | 592.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 09:15:00 | 623.00 | 623.04 | 617.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:45:00 | 622.55 | 623.04 | 617.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 619.10 | 623.65 | 620.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 619.10 | 623.65 | 620.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 622.00 | 623.32 | 620.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 615.65 | 623.32 | 620.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 614.10 | 621.48 | 619.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 614.55 | 621.48 | 619.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 618.15 | 620.81 | 619.53 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 617.10 | 618.52 | 618.67 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 621.85 | 618.87 | 618.76 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 618.25 | 618.67 | 618.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 12:15:00 | 616.55 | 618.25 | 618.49 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 628.95 | 619.78 | 619.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 633.50 | 625.94 | 623.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 629.00 | 629.19 | 626.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 620.60 | 629.19 | 626.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 615.80 | 626.51 | 625.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 615.80 | 626.51 | 625.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 612.80 | 623.77 | 624.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 607.25 | 617.59 | 620.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 571.00 | 566.65 | 575.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 12:45:00 | 570.25 | 566.65 | 575.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 573.15 | 568.52 | 574.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 573.15 | 568.52 | 574.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 572.00 | 569.93 | 574.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 568.15 | 570.09 | 573.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:00:00 | 569.00 | 570.09 | 573.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 13:15:00 | 568.50 | 570.20 | 573.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 539.74 | 544.97 | 551.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 540.55 | 544.97 | 551.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 540.07 | 544.97 | 551.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 11:15:00 | 511.33 | 535.67 | 545.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 547.50 | 541.44 | 540.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 550.45 | 545.86 | 543.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 543.00 | 546.06 | 544.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 543.00 | 546.06 | 544.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 543.00 | 546.06 | 544.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 543.00 | 546.06 | 544.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 543.35 | 545.51 | 544.26 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 540.95 | 543.48 | 543.58 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 556.90 | 545.46 | 544.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 11:15:00 | 559.25 | 550.01 | 546.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 554.80 | 555.21 | 550.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 554.80 | 555.21 | 550.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 554.80 | 555.21 | 550.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 552.90 | 555.21 | 550.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 554.35 | 556.49 | 553.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:45:00 | 557.50 | 556.49 | 553.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 553.15 | 555.82 | 553.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 562.85 | 555.82 | 553.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 14:30:00 | 555.75 | 557.79 | 555.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 549.75 | 555.36 | 555.07 | SL hit (close<static) qty=1.00 sl=550.30 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 541.75 | 552.63 | 553.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 538.80 | 548.20 | 551.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 511.25 | 509.01 | 520.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 512.50 | 509.01 | 520.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 523.10 | 511.83 | 520.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 523.10 | 511.83 | 520.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 518.35 | 513.13 | 520.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 517.10 | 513.13 | 520.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 532.50 | 521.99 | 522.59 | SL hit (close>static) qty=1.00 sl=526.40 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 533.30 | 524.25 | 523.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 544.60 | 532.77 | 528.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 540.30 | 540.88 | 536.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 540.30 | 540.88 | 536.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 536.50 | 539.23 | 536.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:45:00 | 536.35 | 539.23 | 536.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 537.05 | 538.79 | 536.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:15:00 | 538.60 | 538.79 | 536.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:30:00 | 539.10 | 539.65 | 537.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 11:30:00 | 540.50 | 545.87 | 545.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 12:15:00 | 540.40 | 544.78 | 544.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 540.40 | 544.78 | 544.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 539.30 | 543.68 | 544.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 545.70 | 542.99 | 543.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 545.70 | 542.99 | 543.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 545.70 | 542.99 | 543.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 545.70 | 542.99 | 543.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 546.50 | 543.69 | 543.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 547.40 | 543.69 | 543.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 545.80 | 544.11 | 544.09 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 540.25 | 543.95 | 544.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 534.35 | 541.30 | 542.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 515.70 | 510.85 | 517.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 515.70 | 510.85 | 517.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 523.30 | 513.34 | 518.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 524.85 | 513.34 | 518.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 525.00 | 515.67 | 518.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 522.40 | 518.88 | 519.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 523.10 | 516.01 | 515.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 523.10 | 516.01 | 515.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 524.10 | 517.63 | 516.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 534.95 | 536.89 | 531.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 534.95 | 536.89 | 531.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 527.20 | 534.13 | 531.36 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 519.50 | 528.58 | 529.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 517.80 | 526.42 | 528.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 528.30 | 525.79 | 527.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 528.30 | 525.79 | 527.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 528.30 | 525.79 | 527.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 528.30 | 525.79 | 527.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 526.95 | 526.03 | 527.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 513.80 | 526.03 | 527.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 523.75 | 516.69 | 518.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 531.20 | 520.72 | 520.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 531.20 | 520.72 | 520.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 535.55 | 528.07 | 524.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 528.25 | 529.42 | 525.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 528.25 | 529.42 | 525.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 526.15 | 528.77 | 525.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 525.55 | 528.77 | 525.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 526.75 | 528.37 | 526.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 528.25 | 528.37 | 526.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 13:00:00 | 527.90 | 528.27 | 526.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 528.45 | 528.31 | 526.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:45:00 | 528.30 | 528.35 | 526.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 522.40 | 527.15 | 526.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 522.40 | 527.15 | 526.34 | SL hit (close<static) qty=1.00 sl=525.30 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 517.45 | 525.21 | 525.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 515.75 | 523.32 | 524.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 526.50 | 519.89 | 522.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 526.50 | 519.89 | 522.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 526.50 | 519.89 | 522.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 528.00 | 519.89 | 522.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 537.35 | 523.38 | 523.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:45:00 | 540.35 | 523.38 | 523.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 533.90 | 525.49 | 524.35 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 518.40 | 524.67 | 525.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 514.20 | 522.57 | 524.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 515.70 | 515.43 | 519.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 13:15:00 | 515.70 | 515.43 | 519.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 515.70 | 515.43 | 519.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:45:00 | 517.10 | 515.43 | 519.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 508.25 | 505.74 | 508.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 509.25 | 505.74 | 508.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 500.45 | 504.68 | 508.19 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 521.00 | 510.03 | 509.86 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 473.25 | 505.83 | 508.69 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 506.65 | 503.85 | 503.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 528.00 | 508.96 | 505.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 518.80 | 518.90 | 512.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 518.80 | 518.90 | 512.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 519.75 | 519.57 | 514.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 527.00 | 521.69 | 518.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 559.15 | 564.65 | 565.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 559.15 | 564.65 | 565.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 547.55 | 560.65 | 563.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 556.95 | 555.65 | 559.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 556.95 | 555.65 | 559.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 560.15 | 556.85 | 559.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 562.05 | 556.85 | 559.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 558.20 | 557.12 | 559.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 559.45 | 557.12 | 559.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 558.20 | 557.33 | 558.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 559.20 | 557.33 | 558.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 560.50 | 557.97 | 559.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 560.50 | 557.97 | 559.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 559.80 | 558.33 | 559.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 558.95 | 558.33 | 559.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 559.20 | 558.63 | 559.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 558.50 | 558.63 | 559.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 564.10 | 559.70 | 559.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 564.10 | 559.70 | 559.59 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 555.65 | 558.86 | 559.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 552.00 | 557.01 | 558.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 531.90 | 530.95 | 536.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 531.90 | 530.95 | 536.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 512.35 | 510.85 | 515.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 513.00 | 510.85 | 515.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 515.15 | 512.40 | 515.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:45:00 | 516.30 | 512.40 | 515.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 517.00 | 513.32 | 515.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 517.00 | 513.32 | 515.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 518.60 | 514.37 | 516.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 518.60 | 514.37 | 516.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 519.05 | 515.31 | 516.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 519.05 | 515.31 | 516.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 514.55 | 516.00 | 516.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 512.55 | 515.27 | 516.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 506.65 | 513.76 | 515.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 514.05 | 505.47 | 504.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 514.05 | 505.47 | 504.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 516.90 | 509.37 | 506.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 530.15 | 531.18 | 522.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:45:00 | 527.40 | 531.18 | 522.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 518.65 | 528.68 | 522.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 518.65 | 528.68 | 522.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 521.20 | 527.18 | 521.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:30:00 | 522.60 | 526.20 | 521.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 513.15 | 523.59 | 521.18 | SL hit (close<static) qty=1.00 sl=516.65 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 512.00 | 519.52 | 519.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 504.35 | 512.93 | 515.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 458.60 | 456.51 | 468.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 470.10 | 459.23 | 468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 470.10 | 459.23 | 468.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 470.10 | 459.23 | 468.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 472.40 | 461.86 | 468.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 472.40 | 461.86 | 468.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 474.90 | 464.47 | 469.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 474.90 | 464.47 | 469.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 476.35 | 466.84 | 470.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:30:00 | 475.75 | 466.84 | 470.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 484.10 | 473.31 | 472.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 488.30 | 478.27 | 475.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 475.45 | 478.26 | 475.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 13:15:00 | 475.45 | 478.26 | 475.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 475.45 | 478.26 | 475.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 475.45 | 478.26 | 475.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 479.30 | 478.47 | 475.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 490.20 | 478.37 | 476.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:45:00 | 481.55 | 483.17 | 481.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 474.65 | 484.48 | 484.05 | SL hit (close<static) qty=1.00 sl=474.80 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 481.00 | 483.26 | 483.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 476.15 | 481.48 | 482.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 464.00 | 463.97 | 469.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:00:00 | 464.00 | 463.97 | 469.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 432.95 | 427.11 | 433.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 430.30 | 428.84 | 433.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 436.65 | 431.20 | 433.58 | SL hit (close>static) qty=1.00 sl=435.90 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 439.80 | 435.59 | 435.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 443.00 | 438.58 | 436.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 446.60 | 450.13 | 445.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 446.60 | 450.13 | 445.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 446.60 | 450.13 | 445.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 444.35 | 450.13 | 445.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 429.80 | 446.06 | 444.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 429.80 | 446.06 | 444.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 433.25 | 443.50 | 443.31 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 427.50 | 440.30 | 441.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 410.30 | 432.47 | 437.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 421.75 | 414.85 | 424.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 421.75 | 414.85 | 424.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 421.75 | 414.85 | 424.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 424.00 | 414.85 | 424.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 425.00 | 419.00 | 423.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 425.00 | 419.00 | 423.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 424.55 | 420.11 | 423.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 424.55 | 420.11 | 423.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 427.90 | 421.67 | 424.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 427.90 | 421.67 | 424.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 426.65 | 422.66 | 424.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 435.15 | 422.66 | 424.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 437.30 | 427.57 | 426.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 439.10 | 429.88 | 427.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 434.40 | 436.81 | 433.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 434.40 | 436.81 | 433.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 434.60 | 436.37 | 433.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 434.60 | 436.37 | 433.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 432.75 | 435.64 | 433.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 433.35 | 435.64 | 433.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 429.65 | 434.45 | 433.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 429.65 | 434.45 | 433.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 425.10 | 432.58 | 432.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 438.00 | 432.58 | 432.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 436.40 | 433.34 | 432.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 434.90 | 434.17 | 433.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 439.30 | 436.31 | 434.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 425.75 | 435.67 | 434.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 425.10 | 435.67 | 434.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 423.30 | 433.19 | 433.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 423.30 | 433.19 | 433.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 419.75 | 430.51 | 432.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 411.70 | 408.78 | 414.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 411.70 | 408.78 | 414.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 417.05 | 409.99 | 413.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 416.85 | 409.99 | 413.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 414.40 | 410.87 | 413.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:45:00 | 411.90 | 411.95 | 413.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 391.30 | 401.88 | 407.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 389.85 | 389.28 | 395.91 | SL hit (close>ema200) qty=0.50 sl=389.28 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 396.40 | 390.76 | 390.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 400.10 | 393.39 | 391.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 403.65 | 403.83 | 398.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 403.65 | 403.83 | 398.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 402.95 | 404.24 | 400.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 401.55 | 404.24 | 400.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 404.05 | 404.20 | 401.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:15:00 | 402.20 | 404.20 | 401.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 402.20 | 403.80 | 401.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 397.00 | 403.80 | 401.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 396.25 | 402.29 | 400.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 394.00 | 402.29 | 400.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 392.25 | 399.04 | 399.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 390.45 | 396.26 | 398.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 376.80 | 366.87 | 371.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 376.80 | 366.87 | 371.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 376.80 | 366.87 | 371.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 376.80 | 366.87 | 371.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 378.00 | 369.09 | 372.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 378.00 | 369.09 | 372.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 382.50 | 374.39 | 374.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 388.15 | 377.14 | 375.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 406.50 | 409.77 | 403.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 406.50 | 409.77 | 403.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 404.80 | 408.12 | 403.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 404.05 | 408.12 | 403.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 407.35 | 407.37 | 404.42 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 397.80 | 402.74 | 402.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 394.60 | 401.11 | 402.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 399.65 | 398.92 | 400.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 399.65 | 398.92 | 400.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 399.65 | 398.92 | 400.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 399.65 | 398.92 | 400.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 401.00 | 398.66 | 400.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 401.00 | 398.66 | 400.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 401.40 | 399.21 | 400.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:15:00 | 404.10 | 399.21 | 400.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 398.15 | 399.43 | 400.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 401.25 | 399.43 | 400.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 399.55 | 398.71 | 399.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:15:00 | 402.40 | 398.71 | 399.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 402.40 | 399.45 | 399.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 402.40 | 399.45 | 399.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 401.60 | 399.88 | 399.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 402.35 | 399.88 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 407.50 | 401.40 | 400.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 408.00 | 402.72 | 401.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 425.60 | 426.25 | 421.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 09:45:00 | 426.20 | 426.25 | 421.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 431.20 | 428.99 | 425.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 433.50 | 429.83 | 426.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:45:00 | 434.00 | 430.69 | 426.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 434.70 | 430.39 | 427.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 427.05 | 435.68 | 436.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 427.05 | 435.68 | 436.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 425.55 | 433.65 | 435.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 437.35 | 432.20 | 433.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 437.35 | 432.20 | 433.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 437.35 | 432.20 | 433.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 437.35 | 432.20 | 433.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 436.60 | 433.08 | 434.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 434.45 | 433.08 | 434.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 437.95 | 433.96 | 433.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 437.95 | 433.96 | 433.71 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 431.40 | 433.47 | 433.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 429.15 | 432.61 | 433.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 421.55 | 420.13 | 424.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:30:00 | 421.25 | 420.13 | 424.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 423.00 | 420.71 | 424.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 423.85 | 420.71 | 424.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 422.35 | 421.04 | 424.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 422.35 | 421.04 | 424.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 424.10 | 421.65 | 424.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 424.10 | 421.65 | 424.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 425.00 | 422.32 | 424.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 427.80 | 422.32 | 424.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 424.90 | 422.84 | 424.20 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 428.30 | 425.30 | 425.06 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 406.45 | 421.63 | 423.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 405.85 | 413.92 | 418.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 392.90 | 387.82 | 397.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 392.90 | 387.82 | 397.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 392.90 | 387.82 | 397.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 387.75 | 392.63 | 396.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 388.05 | 390.39 | 393.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 389.60 | 390.33 | 393.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 403.10 | 395.85 | 395.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 403.10 | 395.85 | 395.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 415.85 | 402.67 | 399.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 440.20 | 440.77 | 435.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 443.60 | 440.77 | 435.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 433.00 | 438.71 | 435.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 433.00 | 438.71 | 435.77 | SL hit (close<ema400) qty=1.00 sl=435.77 alert=retest1 |

### Cycle 64 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 423.05 | 435.19 | 436.69 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 434.00 | 432.13 | 431.96 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 431.15 | 431.73 | 431.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 428.70 | 430.75 | 431.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 431.65 | 425.61 | 427.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 431.65 | 425.61 | 427.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 431.65 | 425.61 | 427.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 431.65 | 425.61 | 427.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 426.70 | 425.83 | 427.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 423.40 | 425.83 | 427.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 402.23 | 409.17 | 415.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 409.15 | 407.87 | 412.52 | SL hit (close>ema200) qty=0.50 sl=407.87 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 409.60 | 400.39 | 400.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 412.70 | 404.56 | 402.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 406.55 | 406.61 | 403.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 407.15 | 406.61 | 403.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 402.15 | 405.71 | 404.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 401.35 | 405.71 | 404.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 400.40 | 404.65 | 403.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 400.40 | 404.65 | 403.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 402.40 | 403.93 | 403.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 402.40 | 403.93 | 403.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 391.60 | 401.46 | 402.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 385.30 | 396.34 | 399.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 12:15:00 | 393.20 | 392.79 | 396.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 13:00:00 | 393.20 | 392.79 | 396.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 394.35 | 393.22 | 395.77 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 406.30 | 398.24 | 397.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 409.50 | 405.08 | 401.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 406.10 | 406.94 | 403.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 406.10 | 406.94 | 403.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 403.00 | 406.22 | 404.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 401.60 | 406.22 | 404.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 402.30 | 405.44 | 404.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 402.30 | 405.44 | 404.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 400.10 | 404.37 | 403.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 399.65 | 404.37 | 403.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 397.65 | 403.03 | 403.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 396.05 | 401.63 | 402.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 399.15 | 397.12 | 399.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 13:15:00 | 399.15 | 397.12 | 399.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 399.15 | 397.12 | 399.18 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 401.95 | 399.93 | 399.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 15:15:00 | 403.10 | 401.85 | 401.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 403.00 | 405.40 | 403.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 403.00 | 405.40 | 403.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 403.00 | 405.40 | 403.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 403.00 | 405.40 | 403.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 404.80 | 405.28 | 403.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 402.75 | 405.28 | 403.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 403.80 | 404.89 | 404.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:15:00 | 403.35 | 404.89 | 404.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 402.55 | 404.42 | 403.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 402.55 | 404.42 | 403.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 405.15 | 404.57 | 403.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 405.80 | 404.57 | 403.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:45:00 | 405.90 | 405.55 | 404.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 402.50 | 404.16 | 404.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 402.50 | 404.16 | 404.26 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 406.65 | 404.58 | 404.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 409.50 | 405.56 | 404.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 405.10 | 406.47 | 405.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 405.10 | 406.47 | 405.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 405.10 | 406.47 | 405.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 405.10 | 406.47 | 405.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 405.15 | 406.21 | 405.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 404.00 | 406.21 | 405.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 403.05 | 405.01 | 405.11 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 407.50 | 404.57 | 404.44 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 403.30 | 404.32 | 404.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 402.25 | 403.90 | 404.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 401.65 | 401.40 | 402.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 401.65 | 401.40 | 402.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 401.65 | 401.40 | 402.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 401.65 | 401.40 | 402.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 401.75 | 401.47 | 402.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:15:00 | 402.00 | 401.47 | 402.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 401.35 | 401.45 | 402.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 401.00 | 401.39 | 402.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:45:00 | 400.80 | 401.31 | 402.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 404.10 | 401.87 | 402.26 | SL hit (close>static) qty=1.00 sl=402.60 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 404.70 | 402.63 | 402.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 410.25 | 404.60 | 403.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 422.90 | 423.49 | 418.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 422.90 | 423.49 | 418.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 418.55 | 423.03 | 421.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 419.25 | 423.03 | 421.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 419.75 | 422.37 | 421.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:15:00 | 419.55 | 422.37 | 421.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 414.35 | 420.32 | 420.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 412.35 | 418.72 | 419.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 401.80 | 401.16 | 405.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:30:00 | 401.90 | 401.16 | 405.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 401.70 | 401.58 | 403.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 399.85 | 401.16 | 403.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 399.90 | 400.77 | 402.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 394.95 | 393.56 | 393.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 394.95 | 393.56 | 393.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 396.45 | 394.14 | 393.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 399.00 | 399.92 | 397.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 400.05 | 399.92 | 397.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 399.90 | 399.24 | 398.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 400.65 | 399.02 | 398.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 400.60 | 399.42 | 398.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:00:00 | 401.05 | 399.84 | 399.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 397.60 | 401.99 | 402.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 397.60 | 401.99 | 402.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 396.95 | 398.91 | 400.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 394.00 | 393.41 | 395.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 394.00 | 393.41 | 395.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 394.15 | 393.65 | 394.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 394.25 | 393.65 | 394.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 393.25 | 393.57 | 394.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 392.55 | 393.57 | 394.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 392.35 | 392.24 | 393.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 393.00 | 392.25 | 393.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 395.15 | 393.26 | 393.42 | SL hit (close>static) qty=1.00 sl=395.05 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 397.25 | 393.34 | 393.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 401.80 | 398.21 | 396.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 399.20 | 399.86 | 397.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 399.20 | 399.86 | 397.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 399.20 | 399.86 | 397.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 397.90 | 399.86 | 397.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 399.20 | 399.83 | 398.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 399.20 | 399.83 | 398.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 399.00 | 399.67 | 398.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 399.00 | 399.67 | 398.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 398.85 | 399.50 | 398.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 401.20 | 399.40 | 398.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 14:30:00 | 400.10 | 400.54 | 399.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 401.65 | 400.54 | 399.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:00:00 | 404.45 | 401.51 | 400.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 403.00 | 402.02 | 401.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 401.70 | 402.02 | 401.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 401.10 | 401.86 | 401.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 401.10 | 401.86 | 401.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 400.00 | 401.49 | 401.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 397.15 | 401.49 | 401.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 396.90 | 400.57 | 400.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 396.90 | 400.57 | 400.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 393.50 | 397.48 | 398.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 395.80 | 395.17 | 396.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 13:15:00 | 395.80 | 395.17 | 396.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 395.80 | 395.17 | 396.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 395.80 | 395.17 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 399.60 | 396.06 | 397.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 399.70 | 396.06 | 397.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 400.50 | 396.95 | 397.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 404.30 | 396.95 | 397.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 401.25 | 397.81 | 397.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 406.15 | 401.36 | 399.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 401.85 | 402.59 | 400.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:15:00 | 400.25 | 402.59 | 400.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 401.60 | 402.39 | 400.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 402.80 | 401.07 | 400.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 399.20 | 400.71 | 400.56 | SL hit (close<static) qty=1.00 sl=399.60 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 395.60 | 399.69 | 400.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 391.20 | 397.99 | 399.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 396.50 | 395.40 | 397.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 396.50 | 395.40 | 397.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 397.50 | 395.82 | 397.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 397.50 | 395.82 | 397.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 398.20 | 396.29 | 397.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 398.15 | 396.29 | 397.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 400.40 | 397.12 | 397.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 400.40 | 397.12 | 397.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 401.00 | 398.57 | 398.25 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 393.80 | 398.57 | 398.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 387.70 | 394.88 | 396.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 390.75 | 390.75 | 393.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:15:00 | 392.15 | 390.75 | 393.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 393.05 | 391.53 | 392.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:15:00 | 392.45 | 391.53 | 392.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 393.35 | 391.90 | 393.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 393.35 | 391.90 | 393.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 393.05 | 392.13 | 393.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 393.30 | 392.13 | 393.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 394.80 | 392.66 | 393.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 397.40 | 392.66 | 393.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 393.00 | 392.73 | 393.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 392.00 | 392.71 | 393.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 14:15:00 | 394.45 | 393.37 | 393.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 394.45 | 393.37 | 393.32 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 390.50 | 392.90 | 393.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 389.25 | 392.07 | 392.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 384.35 | 383.32 | 385.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 384.35 | 383.32 | 385.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 384.35 | 383.32 | 385.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 384.35 | 383.32 | 385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 384.85 | 383.71 | 385.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 384.85 | 383.71 | 385.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 384.75 | 383.92 | 385.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 384.75 | 383.92 | 385.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 387.50 | 384.64 | 385.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 387.50 | 384.64 | 385.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 386.50 | 385.01 | 385.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 388.80 | 385.01 | 385.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 387.55 | 386.04 | 385.85 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 383.10 | 386.37 | 386.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 381.65 | 384.40 | 385.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 383.45 | 383.16 | 384.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 383.45 | 383.16 | 384.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 383.45 | 383.16 | 384.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 380.20 | 382.66 | 384.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 380.65 | 380.56 | 381.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 380.45 | 380.52 | 381.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:45:00 | 380.40 | 381.31 | 381.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 381.55 | 381.27 | 381.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 381.55 | 381.27 | 381.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 381.00 | 381.22 | 381.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 383.15 | 381.74 | 381.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 383.15 | 381.74 | 381.70 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 380.60 | 381.51 | 381.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 380.10 | 381.23 | 381.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 353.05 | 352.46 | 357.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 354.40 | 352.46 | 357.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 357.00 | 353.84 | 357.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 357.00 | 353.84 | 357.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 357.65 | 354.60 | 357.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 357.65 | 354.60 | 357.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 359.00 | 355.48 | 357.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 359.00 | 355.48 | 357.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 360.60 | 356.51 | 357.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 360.60 | 356.51 | 357.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 367.55 | 359.55 | 358.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 369.90 | 367.13 | 366.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 367.40 | 368.43 | 367.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 367.40 | 368.43 | 367.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 367.40 | 368.43 | 367.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 367.40 | 368.43 | 367.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 367.95 | 368.33 | 367.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 371.30 | 368.05 | 367.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 384.90 | 385.48 | 385.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 384.90 | 385.48 | 385.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 382.05 | 384.80 | 385.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 370.35 | 370.18 | 374.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 370.35 | 370.18 | 374.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 372.60 | 371.15 | 373.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:30:00 | 372.05 | 371.15 | 373.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 372.20 | 371.67 | 373.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 370.15 | 371.26 | 372.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 383.60 | 374.22 | 373.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 383.60 | 374.22 | 373.64 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 375.30 | 377.29 | 377.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 374.35 | 376.70 | 377.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 372.95 | 372.01 | 373.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 375.45 | 372.01 | 373.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 378.10 | 373.23 | 373.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 378.10 | 373.23 | 373.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 375.45 | 373.67 | 374.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:45:00 | 374.40 | 373.93 | 374.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 374.75 | 373.93 | 374.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 374.40 | 374.13 | 374.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 375.80 | 371.82 | 371.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 375.80 | 371.82 | 371.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 378.00 | 373.74 | 372.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 372.70 | 376.25 | 375.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 372.70 | 376.25 | 375.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 372.70 | 376.25 | 375.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 372.40 | 376.25 | 375.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 371.95 | 375.39 | 375.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 371.75 | 375.39 | 375.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 372.40 | 374.79 | 374.84 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 377.00 | 374.76 | 374.67 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 373.75 | 375.73 | 375.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 372.40 | 374.60 | 375.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 372.90 | 371.70 | 372.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 372.90 | 371.70 | 372.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 372.90 | 371.70 | 372.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 372.40 | 371.70 | 372.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 375.30 | 372.42 | 373.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 375.55 | 372.42 | 373.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 371.95 | 372.32 | 373.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 371.30 | 372.32 | 373.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 370.65 | 371.48 | 372.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 381.65 | 372.96 | 372.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 381.65 | 372.96 | 372.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 384.00 | 375.17 | 373.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 379.90 | 380.84 | 377.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:45:00 | 378.90 | 380.84 | 377.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 378.50 | 380.05 | 378.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 378.40 | 380.05 | 378.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 378.30 | 379.70 | 378.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 377.55 | 379.70 | 378.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 378.80 | 379.52 | 378.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 379.75 | 379.52 | 378.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 378.10 | 379.24 | 378.20 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 374.50 | 377.15 | 377.45 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 379.80 | 377.42 | 377.33 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 375.70 | 377.31 | 377.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 372.70 | 376.39 | 376.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 366.00 | 364.97 | 368.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 366.00 | 364.97 | 368.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 366.70 | 365.31 | 367.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 366.90 | 365.31 | 367.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 364.75 | 365.20 | 367.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 368.00 | 365.20 | 367.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 364.95 | 362.03 | 363.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 366.15 | 362.03 | 363.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 363.00 | 362.23 | 363.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:45:00 | 362.00 | 362.24 | 362.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 362.05 | 362.41 | 362.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 361.05 | 362.51 | 362.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:45:00 | 361.20 | 362.26 | 362.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 360.75 | 361.00 | 361.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 362.00 | 361.00 | 361.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 362.80 | 360.05 | 360.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 362.55 | 360.05 | 360.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 361.55 | 360.35 | 360.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 360.90 | 360.35 | 360.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 361.25 | 360.71 | 360.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 361.25 | 360.91 | 360.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 361.05 | 359.70 | 359.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 362.80 | 360.32 | 360.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 362.80 | 360.32 | 360.08 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 359.15 | 360.23 | 360.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 358.00 | 359.46 | 359.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 356.75 | 356.73 | 358.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 356.75 | 356.73 | 358.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 358.20 | 354.50 | 355.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 358.20 | 354.50 | 355.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 358.90 | 355.38 | 356.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 358.95 | 355.38 | 356.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 357.50 | 356.68 | 356.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 360.30 | 357.34 | 356.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 360.40 | 360.98 | 359.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:30:00 | 360.65 | 360.98 | 359.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 360.05 | 360.69 | 360.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 360.00 | 360.69 | 360.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 360.00 | 360.55 | 360.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 359.85 | 360.55 | 360.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 359.50 | 360.34 | 359.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 359.35 | 360.34 | 359.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 358.90 | 360.05 | 359.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 358.70 | 360.05 | 359.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 357.85 | 359.61 | 359.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 356.55 | 358.64 | 359.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 11:15:00 | 358.70 | 358.54 | 359.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 358.70 | 358.54 | 359.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 358.70 | 358.54 | 359.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 358.70 | 358.54 | 359.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 358.70 | 358.57 | 359.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:45:00 | 357.30 | 358.45 | 358.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 339.44 | 344.11 | 347.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 344.20 | 343.76 | 346.85 | SL hit (close>ema200) qty=0.50 sl=343.76 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 338.40 | 337.21 | 337.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 343.20 | 339.25 | 338.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 354.15 | 354.37 | 350.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 354.15 | 354.37 | 350.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 354.00 | 356.34 | 354.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 353.95 | 356.34 | 354.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 354.15 | 355.90 | 354.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 355.70 | 355.55 | 354.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 356.10 | 355.66 | 354.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 352.30 | 354.33 | 354.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 352.30 | 354.33 | 354.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 351.60 | 353.79 | 354.15 | Break + close below crossover candle low |

### Cycle 111 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 359.95 | 354.05 | 354.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 361.00 | 357.76 | 356.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 379.85 | 380.18 | 374.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 379.85 | 380.18 | 374.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 383.30 | 383.99 | 381.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 383.30 | 383.99 | 381.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 377.70 | 382.73 | 381.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 377.70 | 382.73 | 381.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 375.60 | 381.30 | 381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 375.60 | 381.30 | 381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 375.20 | 380.08 | 380.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 373.20 | 378.71 | 379.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 368.25 | 367.38 | 371.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 368.25 | 367.38 | 371.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 370.50 | 368.49 | 370.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 370.50 | 368.49 | 370.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 370.35 | 368.86 | 370.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 369.80 | 368.86 | 370.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 367.80 | 368.65 | 370.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 365.95 | 367.63 | 369.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 365.30 | 367.26 | 369.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 364.85 | 366.78 | 368.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 370.45 | 369.38 | 369.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 370.45 | 369.38 | 369.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 373.10 | 370.25 | 369.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 370.45 | 371.63 | 370.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 370.45 | 371.63 | 370.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 370.45 | 371.63 | 370.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 370.45 | 371.63 | 370.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 371.60 | 371.62 | 370.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 371.05 | 371.62 | 370.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 371.95 | 371.69 | 370.95 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 13:15:00 | 369.40 | 370.60 | 370.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 366.70 | 369.56 | 370.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 365.35 | 358.51 | 361.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 365.35 | 358.51 | 361.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 365.35 | 358.51 | 361.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 365.35 | 358.51 | 361.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 365.55 | 359.92 | 361.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 365.95 | 359.92 | 361.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 365.35 | 362.83 | 362.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 367.40 | 364.27 | 363.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 365.70 | 365.71 | 364.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 365.65 | 365.71 | 364.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 364.35 | 365.44 | 364.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 364.35 | 365.44 | 364.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 364.20 | 365.19 | 364.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 364.20 | 365.19 | 364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 361.50 | 364.45 | 364.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 361.50 | 364.45 | 364.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 362.70 | 364.10 | 363.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 365.00 | 364.10 | 363.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 361.85 | 363.63 | 363.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 361.85 | 363.63 | 363.78 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 369.55 | 364.78 | 364.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 375.30 | 366.89 | 365.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 375.00 | 375.61 | 372.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 15:00:00 | 375.00 | 375.61 | 372.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 367.20 | 373.99 | 372.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 364.40 | 373.99 | 372.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 365.20 | 372.23 | 371.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 365.20 | 372.23 | 371.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 361.25 | 369.08 | 370.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 359.80 | 365.27 | 367.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 364.75 | 364.12 | 366.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 364.75 | 364.12 | 366.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 364.75 | 364.12 | 366.20 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 372.15 | 365.93 | 365.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 378.80 | 371.46 | 368.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 371.95 | 379.15 | 377.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 371.95 | 379.15 | 377.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 371.95 | 379.15 | 377.00 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 370.00 | 374.75 | 375.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 357.10 | 370.20 | 372.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 349.35 | 347.72 | 350.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 349.35 | 347.72 | 350.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 350.50 | 348.57 | 350.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 350.50 | 348.57 | 350.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 351.45 | 349.15 | 350.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 351.20 | 349.15 | 350.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 353.60 | 350.04 | 350.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 353.60 | 350.04 | 350.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 354.80 | 351.64 | 351.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 10:15:00 | 356.30 | 352.57 | 351.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 360.20 | 361.95 | 358.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:45:00 | 360.85 | 361.95 | 358.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 358.90 | 361.34 | 358.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 358.90 | 361.34 | 358.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 356.40 | 360.35 | 358.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 356.40 | 360.35 | 358.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 356.40 | 359.56 | 358.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 355.80 | 359.56 | 358.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 353.40 | 357.78 | 357.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 352.35 | 356.69 | 357.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 10:15:00 | 352.40 | 351.98 | 353.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:00:00 | 352.40 | 351.98 | 353.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 351.90 | 351.96 | 353.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:30:00 | 354.90 | 351.96 | 353.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 353.65 | 352.12 | 353.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 353.65 | 352.12 | 353.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 354.85 | 352.67 | 353.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 355.95 | 352.67 | 353.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 354.80 | 353.55 | 353.52 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 351.40 | 353.12 | 353.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 350.70 | 352.64 | 353.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 353.90 | 352.89 | 353.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 353.90 | 352.89 | 353.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 353.90 | 352.89 | 353.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 353.90 | 352.89 | 353.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 354.05 | 353.12 | 353.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 356.50 | 353.12 | 353.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 354.40 | 353.38 | 353.35 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 351.85 | 353.25 | 353.30 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 353.85 | 353.27 | 353.26 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 350.15 | 352.65 | 352.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 350.00 | 351.46 | 352.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 332.20 | 331.05 | 336.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 10:15:00 | 340.40 | 332.92 | 336.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 340.40 | 332.92 | 336.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 340.40 | 332.92 | 336.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 337.85 | 333.91 | 337.04 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 347.45 | 339.74 | 338.92 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 325.50 | 337.32 | 338.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 321.40 | 328.80 | 333.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 329.05 | 327.76 | 331.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 329.05 | 327.76 | 331.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 329.05 | 327.76 | 331.79 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 338.10 | 333.37 | 333.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 10:15:00 | 340.75 | 335.16 | 334.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 335.75 | 340.23 | 337.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 335.75 | 340.23 | 337.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 335.75 | 340.23 | 337.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 335.75 | 340.23 | 337.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 335.20 | 339.23 | 337.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 336.50 | 339.23 | 337.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 332.70 | 337.39 | 337.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 332.70 | 337.39 | 337.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 333.75 | 336.66 | 336.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 331.70 | 335.67 | 336.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 332.65 | 332.03 | 333.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 332.65 | 332.03 | 333.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 332.65 | 332.03 | 333.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 334.35 | 332.03 | 333.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 332.10 | 331.97 | 333.38 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 337.50 | 334.31 | 334.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 338.80 | 335.21 | 334.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 340.40 | 343.58 | 340.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 340.40 | 343.58 | 340.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 340.40 | 343.58 | 340.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 340.40 | 343.58 | 340.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 338.75 | 342.62 | 340.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 338.75 | 342.62 | 340.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 338.85 | 341.86 | 340.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 338.85 | 341.86 | 340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 335.50 | 340.59 | 339.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 335.50 | 340.59 | 339.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 333.20 | 338.12 | 338.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 331.30 | 334.14 | 336.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 318.25 | 317.57 | 322.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 319.15 | 317.57 | 322.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 326.85 | 320.92 | 322.67 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 328.25 | 324.58 | 324.11 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 320.60 | 323.68 | 324.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 318.90 | 322.19 | 323.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 319.10 | 312.82 | 316.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 319.10 | 312.82 | 316.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 319.10 | 312.82 | 316.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 319.10 | 312.82 | 316.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 320.80 | 314.41 | 316.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 320.80 | 314.41 | 316.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 322.70 | 318.65 | 318.44 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 312.70 | 318.53 | 318.53 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 320.45 | 318.61 | 318.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 323.55 | 319.80 | 319.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 341.30 | 346.92 | 344.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 341.30 | 346.92 | 344.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 341.30 | 346.92 | 344.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 343.50 | 346.92 | 344.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 10:15:00 | 377.85 | 373.78 | 367.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 376.55 | 379.58 | 379.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 374.15 | 378.00 | 378.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 374.50 | 374.33 | 376.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 380.95 | 374.33 | 376.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 378.80 | 375.22 | 376.46 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 378.00 | 377.20 | 377.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 379.35 | 377.79 | 377.49 | Break + close above crossover candle high |

### Cycle 142 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 373.05 | 376.84 | 377.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 361.75 | 373.82 | 375.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 357.55 | 357.27 | 362.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:15:00 | 360.25 | 357.27 | 362.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 356.00 | 355.24 | 357.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:45:00 | 355.45 | 355.44 | 357.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:45:00 | 354.65 | 355.46 | 357.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 358.90 | 355.27 | 356.29 | SL hit (close>static) qty=1.00 sl=357.80 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 363.70 | 357.55 | 357.19 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:15:00 | 534.65 | 2024-05-22 09:15:00 | 538.15 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-05-23 11:30:00 | 536.55 | 2024-05-24 11:15:00 | 553.70 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-05-23 13:00:00 | 537.95 | 2024-05-24 11:15:00 | 553.70 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-05-23 14:15:00 | 537.90 | 2024-05-24 11:15:00 | 553.70 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-05-23 14:45:00 | 538.00 | 2024-05-24 11:15:00 | 553.70 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-06-06 12:30:00 | 490.75 | 2024-06-10 09:15:00 | 508.15 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-06-06 13:45:00 | 491.30 | 2024-06-10 09:15:00 | 508.15 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-06-21 13:15:00 | 513.00 | 2024-06-24 12:15:00 | 523.25 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-06-21 14:30:00 | 513.05 | 2024-06-24 12:15:00 | 523.25 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-06-24 11:15:00 | 513.20 | 2024-06-24 12:15:00 | 523.25 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-07-02 14:30:00 | 540.45 | 2024-07-08 09:15:00 | 594.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 15:15:00 | 539.50 | 2024-07-08 09:15:00 | 593.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 10:30:00 | 585.10 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-08-06 12:15:00 | 585.25 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2024-08-07 09:45:00 | 584.10 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-08-08 09:15:00 | 585.35 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-08-09 14:30:00 | 586.40 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2024-08-12 09:15:00 | 581.35 | 2024-08-16 15:15:00 | 579.80 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-08-22 15:00:00 | 595.90 | 2024-08-23 11:15:00 | 589.80 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-08-23 10:00:00 | 595.25 | 2024-08-23 11:15:00 | 589.80 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-08-26 10:15:00 | 589.05 | 2024-08-27 09:15:00 | 597.25 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-09-13 11:30:00 | 568.15 | 2024-09-19 09:15:00 | 539.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 12:00:00 | 569.00 | 2024-09-19 09:15:00 | 540.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 13:15:00 | 568.50 | 2024-09-19 09:15:00 | 540.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:30:00 | 568.15 | 2024-09-19 11:15:00 | 511.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-13 12:00:00 | 569.00 | 2024-09-19 11:15:00 | 512.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-13 13:15:00 | 568.50 | 2024-09-19 11:15:00 | 511.65 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-01 09:15:00 | 562.85 | 2024-10-03 10:15:00 | 549.75 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-10-01 14:30:00 | 555.75 | 2024-10-03 10:15:00 | 549.75 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-10-08 12:15:00 | 517.10 | 2024-10-09 09:15:00 | 532.50 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-10-11 15:15:00 | 538.60 | 2024-10-17 12:15:00 | 540.40 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-10-14 13:30:00 | 539.10 | 2024-10-17 12:15:00 | 540.40 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-10-17 11:30:00 | 540.50 | 2024-10-17 12:15:00 | 540.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-10-24 14:15:00 | 522.40 | 2024-10-28 12:15:00 | 523.10 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-11-04 09:15:00 | 513.80 | 2024-11-06 09:15:00 | 531.20 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-11-05 15:15:00 | 523.75 | 2024-11-06 09:15:00 | 531.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-11-07 12:15:00 | 528.25 | 2024-11-08 09:15:00 | 522.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-11-07 13:00:00 | 527.90 | 2024-11-08 09:15:00 | 522.40 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-11-07 14:00:00 | 528.45 | 2024-11-08 09:15:00 | 522.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-11-07 14:45:00 | 528.30 | 2024-11-08 09:15:00 | 522.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-27 11:15:00 | 527.00 | 2024-12-12 14:15:00 | 559.15 | STOP_HIT | 1.00 | 6.10% |
| SELL | retest2 | 2024-12-16 14:15:00 | 558.95 | 2024-12-17 09:15:00 | 564.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-12-16 14:45:00 | 559.20 | 2024-12-17 09:15:00 | 564.10 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-16 15:15:00 | 558.50 | 2024-12-17 09:15:00 | 564.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-27 12:45:00 | 512.55 | 2025-01-02 12:15:00 | 514.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-12-27 14:45:00 | 506.65 | 2025-01-02 12:15:00 | 514.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-01-06 12:30:00 | 522.60 | 2025-01-06 13:15:00 | 513.15 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-01-16 09:15:00 | 490.20 | 2025-01-21 10:15:00 | 474.65 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-01-17 11:45:00 | 481.55 | 2025-01-21 10:15:00 | 474.65 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-01-21 11:45:00 | 482.05 | 2025-01-21 12:15:00 | 481.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-01-29 12:45:00 | 430.30 | 2025-01-29 14:15:00 | 436.65 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-02-07 09:15:00 | 438.00 | 2025-02-10 10:15:00 | 423.30 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-02-07 10:00:00 | 436.40 | 2025-02-10 10:15:00 | 423.30 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-02-07 10:30:00 | 434.90 | 2025-02-10 10:15:00 | 423.30 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-07 12:30:00 | 439.30 | 2025-02-10 10:15:00 | 423.30 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-02-13 13:45:00 | 411.90 | 2025-02-14 12:15:00 | 391.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:45:00 | 411.90 | 2025-02-17 14:15:00 | 389.85 | STOP_HIT | 0.50 | 5.35% |
| BUY | retest2 | 2025-03-21 10:30:00 | 433.50 | 2025-03-26 12:15:00 | 427.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-03-21 11:45:00 | 434.00 | 2025-03-26 12:15:00 | 427.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-03-24 09:15:00 | 434.70 | 2025-03-26 12:15:00 | 427.05 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-03-27 11:15:00 | 434.45 | 2025-03-28 11:15:00 | 437.95 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-04-09 10:00:00 | 387.75 | 2025-04-11 11:15:00 | 403.10 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-04-09 13:30:00 | 388.05 | 2025-04-11 11:15:00 | 403.10 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-04-09 15:15:00 | 389.60 | 2025-04-11 11:15:00 | 403.10 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest1 | 2025-04-23 09:15:00 | 443.60 | 2025-04-23 10:15:00 | 433.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-04-23 13:30:00 | 436.85 | 2025-04-25 09:15:00 | 429.65 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-04-23 14:45:00 | 438.45 | 2025-04-25 09:15:00 | 429.65 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-05-02 11:15:00 | 423.40 | 2025-05-07 09:15:00 | 402.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 423.40 | 2025-05-07 13:15:00 | 409.15 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2025-05-27 15:15:00 | 405.80 | 2025-05-28 15:15:00 | 402.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-05-28 11:45:00 | 405.90 | 2025-05-28 15:15:00 | 402.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-04 14:15:00 | 401.00 | 2025-06-05 09:15:00 | 404.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-04 14:45:00 | 400.80 | 2025-06-05 09:15:00 | 404.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-06-17 11:45:00 | 399.85 | 2025-06-23 11:15:00 | 394.95 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2025-06-17 14:15:00 | 399.90 | 2025-06-23 11:15:00 | 394.95 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-06-26 09:15:00 | 400.65 | 2025-07-01 09:15:00 | 397.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-26 12:00:00 | 400.60 | 2025-07-01 09:15:00 | 397.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-26 14:00:00 | 401.05 | 2025-07-01 09:15:00 | 397.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-07 11:15:00 | 392.55 | 2025-07-08 15:15:00 | 395.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-08 10:30:00 | 392.35 | 2025-07-08 15:15:00 | 395.15 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-08 14:00:00 | 393.00 | 2025-07-08 15:15:00 | 395.15 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-09 10:30:00 | 392.70 | 2025-07-10 09:15:00 | 397.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-09 15:15:00 | 391.00 | 2025-07-10 09:15:00 | 397.25 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-15 09:15:00 | 401.20 | 2025-07-21 09:15:00 | 396.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-15 14:30:00 | 400.10 | 2025-07-21 09:15:00 | 396.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-15 15:00:00 | 401.65 | 2025-07-21 09:15:00 | 396.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-18 11:00:00 | 404.45 | 2025-07-21 09:15:00 | 396.90 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-28 10:15:00 | 402.80 | 2025-07-28 11:15:00 | 399.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-05 10:45:00 | 392.00 | 2025-08-05 14:15:00 | 394.45 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-08-18 10:45:00 | 380.20 | 2025-08-21 10:15:00 | 383.15 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-08-19 13:15:00 | 380.65 | 2025-08-21 10:15:00 | 383.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-19 13:45:00 | 380.45 | 2025-08-21 10:15:00 | 383.15 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-20 11:45:00 | 380.40 | 2025-08-21 10:15:00 | 383.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-10 09:15:00 | 371.30 | 2025-09-24 13:15:00 | 384.90 | STOP_HIT | 1.00 | 3.66% |
| SELL | retest2 | 2025-09-30 13:30:00 | 370.15 | 2025-10-01 09:15:00 | 383.60 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-10-10 11:45:00 | 374.40 | 2025-10-15 10:15:00 | 375.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-10-10 12:15:00 | 374.75 | 2025-10-15 10:15:00 | 375.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-10 13:30:00 | 374.40 | 2025-10-15 10:15:00 | 375.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-10-28 10:15:00 | 371.30 | 2025-10-29 10:15:00 | 381.65 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-10-28 11:30:00 | 370.65 | 2025-10-29 10:15:00 | 381.65 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-11-12 11:45:00 | 362.00 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-12 15:00:00 | 362.05 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-11-13 12:30:00 | 361.05 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-13 13:45:00 | 361.20 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-17 11:15:00 | 360.90 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-17 13:00:00 | 361.25 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-11-17 15:15:00 | 361.25 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-11-20 09:45:00 | 361.05 | 2025-11-20 10:15:00 | 362.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-02 14:45:00 | 357.30 | 2025-12-09 09:15:00 | 339.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 14:45:00 | 357.30 | 2025-12-09 11:15:00 | 344.20 | STOP_HIT | 0.50 | 3.67% |
| BUY | retest2 | 2025-12-29 14:15:00 | 355.70 | 2025-12-30 11:15:00 | 352.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-29 15:00:00 | 356.10 | 2025-12-30 11:15:00 | 352.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-13 11:45:00 | 365.95 | 2026-01-14 13:15:00 | 370.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-13 12:45:00 | 365.30 | 2026-01-14 13:15:00 | 370.45 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-13 14:00:00 | 364.85 | 2026-01-14 13:15:00 | 370.45 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-27 09:15:00 | 365.00 | 2026-01-27 10:15:00 | 361.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-04-13 10:15:00 | 343.50 | 2026-04-20 10:15:00 | 377.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 14:45:00 | 355.45 | 2026-05-06 14:15:00 | 358.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-05-06 09:45:00 | 354.65 | 2026-05-06 14:15:00 | 358.90 | STOP_HIT | 1.00 | -1.20% |
