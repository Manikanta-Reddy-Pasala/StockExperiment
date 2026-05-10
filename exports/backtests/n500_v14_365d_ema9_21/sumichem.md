# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 485.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 19 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 69 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 80 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 52
- **Target hits / Stop hits / Partials:** 0 / 72 / 8
- **Avg / median % per leg:** 0.22% / -0.68%
- **Sum % (uncompounded):** 17.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 10 | 30.3% | 0 | 33 | 0 | -0.57% | -18.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.40% | -4.8% |
| BUY @ 3rd Alert (retest2) | 31 | 10 | 32.3% | 0 | 31 | 0 | -0.45% | -14.0% |
| SELL (all) | 47 | 18 | 38.3% | 0 | 39 | 8 | 0.78% | 36.6% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.10% | 0.1% |
| SELL @ 3rd Alert (retest2) | 46 | 17 | 37.0% | 0 | 38 | 8 | 0.79% | 36.5% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.56% | -4.7% |
| retest2 (combined) | 77 | 27 | 35.1% | 0 | 69 | 8 | 0.29% | 22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 518.80 | 508.99 | 508.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 522.20 | 516.68 | 513.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 525.15 | 525.27 | 521.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 525.40 | 525.27 | 521.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 526.60 | 525.84 | 522.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 526.60 | 525.84 | 522.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 520.45 | 524.46 | 523.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 520.45 | 524.46 | 523.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 522.25 | 524.02 | 522.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 525.75 | 524.02 | 522.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 523.90 | 524.99 | 523.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 533.35 | 534.63 | 534.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 533.35 | 534.63 | 534.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 533.35 | 534.63 | 534.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 531.60 | 534.02 | 534.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 534.35 | 533.24 | 533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 532.75 | 533.14 | 533.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 525.30 | 530.94 | 532.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 527.85 | 530.32 | 531.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 514.10 | 531.31 | 531.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 501.46 | 510.32 | 518.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 513.00 | 510.65 | 517.41 | SL hit (close>ema200) qty=0.50 sl=510.65 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:15:00 | 499.03 | 509.71 | 514.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 510.65 | 505.30 | 508.90 | SL hit (close>ema200) qty=0.50 sl=505.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 513.25 | 506.13 | 505.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 513.25 | 506.13 | 505.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 516.15 | 508.13 | 506.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 515.45 | 517.01 | 513.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 515.45 | 517.01 | 513.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 519.05 | 517.16 | 514.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:30:00 | 514.65 | 517.16 | 514.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 515.55 | 516.97 | 514.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 515.85 | 516.97 | 514.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 513.10 | 516.19 | 514.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 513.10 | 516.19 | 514.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 512.10 | 515.38 | 514.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 512.20 | 515.38 | 514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 510.45 | 513.36 | 513.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 509.45 | 512.32 | 513.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 495.00 | 494.92 | 499.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 497.05 | 494.92 | 499.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 495.00 | 494.94 | 499.00 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 500.20 | 498.15 | 498.14 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 495.45 | 497.61 | 497.90 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 13:15:00 | 499.00 | 498.18 | 498.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 500.35 | 498.62 | 498.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 495.45 | 499.22 | 499.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 492.10 | 497.79 | 498.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 499.70 | 496.21 | 497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 505.50 | 498.06 | 498.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 505.95 | 498.06 | 498.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 509.05 | 500.26 | 499.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 512.00 | 502.61 | 500.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 514.80 | 515.31 | 510.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:00:00 | 514.80 | 515.31 | 510.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 512.30 | 513.66 | 511.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 512.30 | 513.66 | 511.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 511.00 | 513.13 | 511.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 511.75 | 513.13 | 511.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 513.65 | 513.23 | 511.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 516.15 | 512.85 | 511.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 516.25 | 514.17 | 512.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 516.40 | 514.69 | 512.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 516.15 | 514.89 | 513.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 527.40 | 522.50 | 519.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 523.50 | 522.50 | 519.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 520.00 | 524.59 | 522.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 520.00 | 524.59 | 522.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 517.40 | 523.15 | 521.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 517.40 | 523.15 | 521.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 525.65 | 521.18 | 520.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 532.30 | 525.32 | 523.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 532.80 | 533.47 | 530.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:45:00 | 533.10 | 533.47 | 530.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 542.45 | 535.27 | 531.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 545.10 | 535.27 | 531.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:45:00 | 543.85 | 547.89 | 545.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 545.25 | 547.38 | 545.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 555.35 | 544.29 | 542.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 559.85 | 547.40 | 544.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 601.45 | 602.53 | 593.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 601.45 | 602.53 | 593.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 594.95 | 600.82 | 594.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 594.95 | 600.82 | 594.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 585.45 | 597.74 | 593.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 585.45 | 597.74 | 593.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 585.00 | 595.20 | 592.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 585.00 | 595.20 | 592.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 592.45 | 593.03 | 592.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 595.60 | 593.18 | 592.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 595.00 | 593.52 | 592.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 589.25 | 592.40 | 592.38 | SL hit (close<static) qty=1.00 sl=590.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 589.25 | 592.40 | 592.38 | SL hit (close<static) qty=1.00 sl=590.40 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 586.75 | 591.27 | 591.87 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 595.85 | 592.14 | 592.01 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 588.10 | 591.91 | 592.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 584.75 | 590.48 | 591.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 586.55 | 586.05 | 588.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 588.75 | 586.59 | 588.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:15:00 | 586.00 | 586.54 | 587.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 605.30 | 590.81 | 588.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 605.30 | 590.81 | 588.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 632.65 | 603.55 | 595.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 642.60 | 644.50 | 631.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:15:00 | 652.70 | 644.50 | 631.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:45:00 | 652.60 | 647.63 | 635.64 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 637.00 | 645.28 | 637.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 637.00 | 645.28 | 637.55 | SL hit (close<ema400) qty=1.00 sl=637.55 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 637.00 | 645.28 | 637.55 | SL hit (close<ema400) qty=1.00 sl=637.55 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 637.00 | 645.28 | 637.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 628.50 | 641.93 | 636.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 628.50 | 641.93 | 636.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 629.95 | 639.53 | 636.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 649.40 | 639.53 | 636.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:30:00 | 637.30 | 636.02 | 635.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 646.00 | 636.02 | 635.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 635.75 | 636.31 | 635.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 628.95 | 632.41 | 633.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 633.30 | 632.58 | 633.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 630.85 | 632.24 | 633.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 624.00 | 632.24 | 633.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 616.00 | 628.99 | 631.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 613.40 | 628.99 | 631.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 582.73 | 593.71 | 605.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 583.40 | 581.89 | 591.35 | SL hit (close>ema200) qty=0.50 sl=581.89 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 596.00 | 589.46 | 589.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 598.25 | 593.81 | 591.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 592.80 | 594.28 | 592.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 12:00:00 | 592.80 | 594.28 | 592.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 590.55 | 593.54 | 592.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 590.55 | 593.54 | 592.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 590.95 | 593.02 | 592.04 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 585.30 | 590.38 | 590.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 582.95 | 589.01 | 590.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 583.30 | 582.87 | 586.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 583.30 | 582.87 | 586.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 585.60 | 583.41 | 586.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 585.60 | 583.41 | 586.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 586.40 | 584.01 | 586.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 586.40 | 584.01 | 586.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 584.60 | 584.13 | 586.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 582.85 | 584.13 | 586.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 14:15:00 | 586.80 | 584.35 | 585.78 | SL hit (close>static) qty=1.00 sl=586.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 582.50 | 584.35 | 585.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 587.40 | 584.66 | 585.65 | SL hit (close>static) qty=1.00 sl=586.40 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 594.65 | 586.66 | 586.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 598.65 | 589.06 | 587.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 588.25 | 590.85 | 589.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 592.90 | 591.26 | 589.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 588.95 | 591.26 | 589.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 588.55 | 590.72 | 589.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 585.85 | 590.72 | 589.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 589.80 | 590.53 | 589.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 589.00 | 590.53 | 589.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 594.85 | 591.40 | 589.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 600.70 | 594.54 | 592.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 599.60 | 595.59 | 592.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 599.75 | 597.47 | 595.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 600.00 | 597.85 | 595.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 594.75 | 597.23 | 595.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 594.75 | 597.23 | 595.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 595.00 | 596.78 | 595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 595.65 | 596.78 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 594.40 | 596.31 | 595.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 595.00 | 596.31 | 595.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 586.00 | 592.84 | 593.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 564.80 | 564.03 | 569.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 564.80 | 564.03 | 569.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 566.85 | 565.02 | 568.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 567.50 | 565.02 | 568.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 568.95 | 565.80 | 568.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 563.60 | 566.93 | 568.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 576.95 | 569.38 | 569.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 576.95 | 569.38 | 569.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 586.00 | 573.70 | 571.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 583.60 | 585.32 | 579.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 583.60 | 585.32 | 579.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 586.00 | 585.46 | 580.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 581.55 | 585.46 | 580.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 581.30 | 583.71 | 581.76 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 574.40 | 580.50 | 580.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 572.05 | 578.81 | 579.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 572.70 | 572.11 | 575.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 572.70 | 572.11 | 575.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 572.75 | 572.23 | 575.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 572.75 | 572.23 | 575.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 568.30 | 570.14 | 573.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 565.55 | 567.96 | 571.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 565.90 | 567.96 | 571.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 565.65 | 567.59 | 570.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 570.35 | 571.11 | 571.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 564.65 | 569.29 | 570.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 556.35 | 555.25 | 559.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:30:00 | 555.15 | 555.25 | 559.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 557.55 | 555.71 | 558.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 557.00 | 555.71 | 558.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 558.50 | 556.27 | 558.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 558.50 | 556.27 | 558.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 556.75 | 556.36 | 558.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 555.70 | 556.36 | 558.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 555.45 | 556.34 | 558.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 556.35 | 555.53 | 557.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 561.20 | 557.06 | 557.86 | SL hit (close>static) qty=1.00 sl=558.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 561.20 | 557.06 | 557.86 | SL hit (close>static) qty=1.00 sl=558.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 561.20 | 557.06 | 557.86 | SL hit (close>static) qty=1.00 sl=558.95 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 561.15 | 558.60 | 558.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 570.10 | 560.90 | 559.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 559.95 | 561.94 | 562.14 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 599.95 | 569.54 | 565.57 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 569.35 | 573.77 | 574.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 564.10 | 571.83 | 573.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 534.95 | 534.58 | 540.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:45:00 | 534.10 | 534.58 | 540.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 536.00 | 534.02 | 538.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 537.80 | 534.02 | 538.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 538.30 | 535.53 | 537.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 538.30 | 535.53 | 537.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 544.50 | 537.33 | 538.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 544.50 | 537.33 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 544.00 | 538.66 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 540.15 | 538.66 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 535.55 | 538.07 | 538.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 535.00 | 538.07 | 538.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 534.75 | 537.56 | 538.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 541.80 | 538.34 | 538.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 541.80 | 538.34 | 538.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 541.80 | 538.34 | 538.25 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 535.55 | 538.51 | 538.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 530.85 | 536.98 | 537.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 534.60 | 534.54 | 536.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 530.80 | 534.54 | 536.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 524.05 | 528.27 | 531.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 523.50 | 527.62 | 530.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 523.20 | 525.36 | 527.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 530.25 | 526.34 | 528.07 | SL hit (close>ema400) qty=1.00 sl=528.07 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 523.00 | 527.04 | 527.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 523.25 | 525.43 | 526.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 524.05 | 522.39 | 524.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 525.25 | 522.39 | 524.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 524.50 | 522.81 | 524.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 525.05 | 522.81 | 524.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 527.05 | 523.66 | 524.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 527.45 | 523.66 | 524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 521.50 | 523.23 | 524.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 523.75 | 523.23 | 524.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 521.20 | 521.42 | 522.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 516.00 | 521.62 | 522.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 517.15 | 519.66 | 521.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 516.30 | 519.66 | 521.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 525.30 | 520.36 | 521.38 | SL hit (close>static) qty=1.00 sl=523.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 525.30 | 520.36 | 521.38 | SL hit (close>static) qty=1.00 sl=523.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 525.30 | 520.36 | 521.38 | SL hit (close>static) qty=1.00 sl=523.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 518.10 | 521.82 | 522.18 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 525.05 | 522.33 | 522.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 528.80 | 523.71 | 522.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 520.00 | 522.91 | 523.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 518.60 | 522.05 | 522.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 519.20 | 518.68 | 520.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:45:00 | 518.80 | 518.68 | 520.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 525.80 | 520.12 | 520.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 514.55 | 519.59 | 520.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 516.80 | 519.59 | 520.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 511.15 | 518.08 | 519.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 490.96 | 499.82 | 502.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 488.82 | 495.87 | 499.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 485.59 | 495.87 | 499.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 477.55 | 477.52 | 482.96 | SL hit (close>ema200) qty=0.50 sl=477.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 477.55 | 477.52 | 482.96 | SL hit (close>ema200) qty=0.50 sl=477.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 477.55 | 477.52 | 482.96 | SL hit (close>ema200) qty=0.50 sl=477.52 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 455.00 | 451.60 | 451.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 455.50 | 452.38 | 451.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 452.70 | 453.81 | 452.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 451.80 | 453.41 | 452.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 451.80 | 453.41 | 452.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 456.45 | 454.02 | 453.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 452.10 | 454.02 | 453.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 455.00 | 454.21 | 453.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 453.65 | 454.21 | 453.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 456.80 | 454.73 | 453.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 453.90 | 454.73 | 453.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 461.00 | 468.40 | 465.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 461.00 | 468.40 | 465.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 459.70 | 466.66 | 464.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 459.70 | 466.66 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 460.40 | 463.10 | 463.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 458.10 | 461.60 | 462.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 463.25 | 461.65 | 462.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 463.65 | 462.05 | 462.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 464.80 | 462.05 | 462.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 462.40 | 462.30 | 462.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:30:00 | 463.50 | 462.30 | 462.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 462.00 | 462.24 | 462.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 458.20 | 462.24 | 462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 466.25 | 460.76 | 461.32 | SL hit (close>static) qty=1.00 sl=463.15 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 458.05 | 460.89 | 461.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 466.00 | 461.85 | 461.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 466.00 | 461.85 | 461.65 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 457.05 | 461.46 | 461.79 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 462.35 | 458.40 | 457.90 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 453.60 | 457.19 | 457.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 453.50 | 456.46 | 457.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 455.10 | 455.26 | 456.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 456.20 | 455.44 | 456.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 456.40 | 455.44 | 456.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 455.80 | 455.52 | 456.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 458.50 | 455.52 | 456.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 453.55 | 455.12 | 455.91 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 458.10 | 455.50 | 455.36 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 451.90 | 454.81 | 455.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 450.25 | 453.89 | 454.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 442.75 | 441.22 | 445.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 442.75 | 441.22 | 445.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 450.40 | 443.33 | 445.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 449.30 | 443.33 | 445.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 451.85 | 445.04 | 446.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 451.85 | 445.04 | 446.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 450.70 | 447.08 | 446.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 453.90 | 448.44 | 447.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 456.00 | 456.22 | 453.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 456.00 | 456.22 | 453.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 459.35 | 459.05 | 456.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 463.30 | 460.26 | 457.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 463.80 | 461.79 | 459.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 464.50 | 462.03 | 459.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 463.75 | 463.57 | 461.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 472.85 | 466.01 | 463.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:15:00 | 473.25 | 466.01 | 463.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 462.70 | 465.35 | 463.29 | SL hit (close<static) qty=1.00 sl=463.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 475.45 | 468.75 | 467.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 474.20 | 473.30 | 472.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 462.20 | 467.43 | 469.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 439.60 | 439.49 | 444.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 436.15 | 439.49 | 444.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 442.00 | 439.21 | 442.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 444.00 | 439.21 | 442.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 444.90 | 440.35 | 442.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 438.45 | 439.24 | 441.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 416.53 | 422.34 | 426.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 419.00 | 417.59 | 421.72 | SL hit (close>ema200) qty=0.50 sl=417.59 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 434.85 | 424.09 | 423.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 440.65 | 427.40 | 425.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 429.80 | 430.43 | 427.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 429.80 | 430.43 | 427.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 428.90 | 430.95 | 428.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 427.90 | 430.95 | 428.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 420.00 | 428.76 | 427.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 420.00 | 428.76 | 427.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 416.80 | 426.37 | 426.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 410.00 | 423.09 | 425.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 405.55 | 403.68 | 408.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 405.55 | 403.68 | 408.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 407.30 | 404.15 | 407.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 407.65 | 404.15 | 407.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 408.60 | 405.04 | 407.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 404.10 | 405.04 | 407.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 408.70 | 405.77 | 407.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 408.70 | 405.77 | 407.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 409.00 | 406.42 | 407.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 409.00 | 406.42 | 407.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 408.55 | 406.84 | 408.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 408.30 | 406.84 | 408.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 408.25 | 407.13 | 408.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 408.25 | 407.13 | 408.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 410.00 | 407.70 | 408.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 409.10 | 407.70 | 408.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 408.40 | 407.84 | 408.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 409.90 | 407.84 | 408.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 409.20 | 408.11 | 408.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 411.05 | 408.11 | 408.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 414.45 | 409.38 | 408.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 416.15 | 410.73 | 409.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 408.70 | 412.98 | 411.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 411.40 | 412.67 | 411.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:15:00 | 413.05 | 412.67 | 411.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 413.50 | 412.43 | 411.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 412.45 | 412.61 | 411.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 412.55 | 419.34 | 420.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 422.70 | 417.29 | 417.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 425.00 | 421.35 | 419.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 427.85 | 428.58 | 424.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 427.85 | 428.58 | 424.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 425.50 | 427.77 | 424.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 419.60 | 427.77 | 424.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 419.95 | 426.21 | 424.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 419.05 | 426.21 | 424.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 418.50 | 424.66 | 423.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 418.85 | 424.66 | 423.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 417.20 | 423.17 | 423.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 416.15 | 421.77 | 422.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 408.05 | 407.76 | 411.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 409.00 | 407.76 | 411.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 406.95 | 407.81 | 410.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 405.15 | 407.81 | 410.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 413.00 | 408.84 | 410.97 | SL hit (close>static) qty=1.00 sl=410.95 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 413.10 | 410.69 | 410.64 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 409.85 | 410.69 | 410.80 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 412.15 | 410.98 | 410.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 15:15:00 | 412.90 | 411.36 | 411.10 | Break + close above crossover candle high |

### Cycle 56 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 407.30 | 410.55 | 410.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 405.05 | 408.49 | 409.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 407.60 | 407.90 | 409.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 410.85 | 408.49 | 409.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 410.85 | 408.49 | 409.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 411.50 | 409.09 | 409.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 411.50 | 409.09 | 409.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 413.60 | 409.99 | 409.84 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 407.00 | 409.67 | 409.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 405.85 | 408.91 | 409.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 406.60 | 406.26 | 407.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 406.60 | 406.26 | 407.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 400.40 | 405.09 | 406.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:15:00 | 398.20 | 405.09 | 406.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 407.50 | 404.63 | 405.68 | SL hit (close>static) qty=1.00 sl=406.85 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 399.75 | 404.14 | 405.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:30:00 | 399.30 | 402.52 | 404.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 399.60 | 401.93 | 403.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 396.75 | 396.71 | 398.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:30:00 | 398.00 | 396.71 | 398.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 395.70 | 396.07 | 397.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 393.40 | 396.07 | 397.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:15:00 | 393.70 | 396.11 | 397.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 395.45 | 397.74 | 397.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 387.20 | 395.64 | 396.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 387.45 | 386.75 | 390.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 387.45 | 386.75 | 390.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 390.05 | 387.41 | 390.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 390.05 | 387.41 | 390.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 391.75 | 388.28 | 390.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 391.75 | 388.28 | 390.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 393.10 | 389.24 | 390.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 392.40 | 389.24 | 390.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 397.10 | 391.98 | 391.78 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 390.85 | 391.59 | 391.65 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 393.15 | 391.90 | 391.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 394.05 | 392.33 | 391.99 | Break + close above crossover candle high |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 386.85 | 391.34 | 391.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 381.05 | 388.50 | 390.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 382.50 | 380.22 | 383.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:00:00 | 382.50 | 380.22 | 383.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 386.00 | 381.38 | 383.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 386.00 | 381.38 | 383.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 386.00 | 382.30 | 383.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 387.85 | 382.30 | 383.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 393.10 | 385.89 | 385.25 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 381.40 | 386.75 | 387.37 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 387.65 | 385.84 | 385.77 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 380.80 | 385.02 | 385.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 375.80 | 381.98 | 383.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 377.90 | 376.65 | 379.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 377.90 | 376.65 | 379.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 380.65 | 377.45 | 379.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 380.65 | 377.45 | 379.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 379.45 | 377.85 | 379.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 377.75 | 378.15 | 379.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 387.05 | 380.16 | 380.34 | SL hit (close>static) qty=1.00 sl=381.40 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 384.55 | 381.04 | 380.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 387.85 | 384.51 | 382.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 373.85 | 382.79 | 382.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 375.15 | 381.26 | 381.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 368.90 | 375.48 | 378.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 380.40 | 375.67 | 375.49 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 373.25 | 375.49 | 375.58 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 377.60 | 375.91 | 375.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 383.15 | 377.36 | 376.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 424.15 | 426.68 | 423.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 11:45:00 | 424.25 | 426.68 | 423.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 423.65 | 426.07 | 423.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:45:00 | 421.25 | 426.07 | 423.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 425.85 | 426.03 | 423.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 426.50 | 425.95 | 423.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 426.00 | 425.95 | 423.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 439.20 | 443.35 | 443.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 439.20 | 443.35 | 443.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 439.20 | 443.35 | 443.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 432.70 | 441.22 | 442.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 443.40 | 439.95 | 441.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 444.90 | 440.94 | 441.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 445.00 | 440.94 | 441.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 446.80 | 443.11 | 442.77 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 438.95 | 442.26 | 442.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 435.15 | 439.99 | 441.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 442.15 | 437.09 | 439.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 440.25 | 437.72 | 439.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 438.30 | 437.62 | 438.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 15:15:00 | 416.38 | 426.02 | 431.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 429.40 | 426.70 | 431.06 | SL hit (close>ema200) qty=0.50 sl=426.70 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 449.65 | 433.96 | 433.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 455.00 | 440.75 | 436.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 491.40 | 496.02 | 486.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:15:00 | 489.50 | 496.02 | 486.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 487.00 | 493.01 | 486.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 487.15 | 493.01 | 486.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 486.25 | 490.90 | 486.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 486.25 | 490.90 | 486.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 485.10 | 489.74 | 486.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 485.10 | 489.74 | 486.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 485.90 | 488.97 | 486.47 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 09:15:00 | 525.75 | 2025-05-22 12:15:00 | 533.35 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-05-16 11:30:00 | 523.90 | 2025-05-22 12:15:00 | 533.35 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-05-26 09:30:00 | 525.30 | 2025-05-27 15:15:00 | 501.46 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2025-05-26 09:30:00 | 525.30 | 2025-05-28 10:15:00 | 513.00 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2025-05-26 11:00:00 | 527.85 | 2025-05-29 09:15:00 | 499.03 | PARTIAL | 0.50 | 5.46% |
| SELL | retest2 | 2025-05-26 11:00:00 | 527.85 | 2025-05-30 09:15:00 | 510.65 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2025-05-27 09:15:00 | 514.10 | 2025-06-05 13:15:00 | 513.25 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-06-26 11:15:00 | 516.15 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-06-26 13:00:00 | 516.25 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-06-26 14:30:00 | 516.40 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-27 09:15:00 | 516.15 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-07-07 12:15:00 | 545.10 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-09 13:45:00 | 543.85 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-09 14:30:00 | 545.25 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-22 11:30:00 | 595.60 | 2025-07-23 12:15:00 | 589.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-23 09:45:00 | 595.00 | 2025-07-23 12:15:00 | 589.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-28 13:15:00 | 586.00 | 2025-07-29 13:15:00 | 605.30 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2025-08-01 09:15:00 | 652.70 | 2025-08-01 13:15:00 | 637.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2025-08-01 10:45:00 | 652.60 | 2025-08-01 13:15:00 | 637.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-08-04 09:15:00 | 649.40 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-08-04 12:30:00 | 637.30 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-04 13:15:00 | 646.00 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-05 10:15:00 | 635.75 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-06 10:15:00 | 613.40 | 2025-08-08 09:15:00 | 582.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 10:15:00 | 613.40 | 2025-08-11 10:15:00 | 583.40 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-08-19 13:15:00 | 582.85 | 2025-08-19 14:15:00 | 586.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-08-19 15:15:00 | 582.50 | 2025-08-20 09:15:00 | 587.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-22 09:45:00 | 600.70 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-08-22 12:15:00 | 599.60 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-25 09:30:00 | 599.75 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-08-25 10:45:00 | 600.00 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-02 09:15:00 | 563.60 | 2025-09-02 10:15:00 | 576.95 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-09-09 13:30:00 | 565.55 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-09 14:00:00 | 565.90 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-09 14:45:00 | 565.65 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-16 13:15:00 | 555.70 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-16 13:45:00 | 555.45 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-17 09:45:00 | 556.35 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-03 11:15:00 | 535.00 | 2025-10-06 09:15:00 | 541.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-03 12:15:00 | 534.75 | 2025-10-06 09:15:00 | 541.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest1 | 2025-10-08 09:15:00 | 530.80 | 2025-10-10 11:15:00 | 530.25 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-10-09 10:30:00 | 523.50 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-10 11:00:00 | 523.20 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-13 15:15:00 | 523.00 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-14 09:45:00 | 523.25 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-16 15:15:00 | 516.00 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-10-17 09:30:00 | 517.15 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-10-17 10:00:00 | 516.30 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-10-27 12:45:00 | 514.55 | 2025-11-06 09:15:00 | 490.96 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2025-10-27 13:15:00 | 516.80 | 2025-11-06 11:15:00 | 488.82 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-10-28 09:15:00 | 511.15 | 2025-11-06 11:15:00 | 485.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 12:45:00 | 514.55 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2025-10-27 13:15:00 | 516.80 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 7.59% |
| SELL | retest2 | 2025-10-28 09:15:00 | 511.15 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2025-12-04 09:15:00 | 458.20 | 2025-12-04 14:15:00 | 466.25 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-05 09:15:00 | 458.05 | 2025-12-05 11:15:00 | 466.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-26 13:30:00 | 463.30 | 2025-12-30 14:15:00 | 462.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-12-29 09:30:00 | 463.80 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-12-29 11:15:00 | 464.50 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-12-30 10:30:00 | 463.75 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-30 14:15:00 | 473.25 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-01-05 10:15:00 | 475.45 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-01-06 14:45:00 | 474.20 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-01-14 12:30:00 | 438.45 | 2026-01-21 09:15:00 | 416.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 438.45 | 2026-01-21 15:15:00 | 419.00 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2026-02-02 11:15:00 | 413.05 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2026-02-02 13:30:00 | 413.50 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2026-02-02 14:45:00 | 412.45 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2026-02-17 09:15:00 | 405.15 | 2026-02-17 09:15:00 | 413.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-25 13:15:00 | 398.20 | 2026-02-26 10:15:00 | 407.50 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-02-27 10:45:00 | 399.75 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-27 12:30:00 | 399.30 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-27 14:00:00 | 399.60 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-03-05 10:15:00 | 393.40 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-05 14:15:00 | 393.70 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-24 14:30:00 | 377.75 | 2026-03-25 09:15:00 | 387.05 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-04-16 14:30:00 | 426.50 | 2026-04-24 12:15:00 | 439.20 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2026-04-16 15:15:00 | 426.00 | 2026-04-24 12:15:00 | 439.20 | STOP_HIT | 1.00 | 3.10% |
| SELL | retest2 | 2026-04-29 13:30:00 | 438.30 | 2026-04-30 15:15:00 | 416.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:30:00 | 438.30 | 2026-05-04 09:15:00 | 429.40 | STOP_HIT | 0.50 | 2.03% |
