# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 631.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 52 |
| ALERT2 | 52 |
| ALERT2_SKIP | 27 |
| ALERT3 | 127 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 61 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 40
- **Target hits / Stop hits / Partials:** 2 / 63 / 7
- **Avg / median % per leg:** 0.39% / -0.44%
- **Sum % (uncompounded):** 27.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 16 | 47.1% | 2 | 32 | 0 | 0.61% | 20.9% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.64% | -2.6% |
| BUY @ 3rd Alert (retest2) | 30 | 16 | 53.3% | 2 | 28 | 0 | 0.78% | 23.5% |
| SELL (all) | 38 | 16 | 42.1% | 0 | 31 | 7 | 0.18% | 7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 16 | 42.1% | 0 | 31 | 7 | 0.18% | 7.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.64% | -2.6% |
| retest2 (combined) | 68 | 32 | 47.1% | 2 | 59 | 7 | 0.45% | 30.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 433.20 | 426.82 | 426.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 440.80 | 430.65 | 427.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 462.80 | 463.39 | 458.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 460.50 | 463.39 | 458.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 459.75 | 462.66 | 458.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 458.85 | 462.66 | 458.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 460.70 | 462.27 | 458.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:45:00 | 461.95 | 461.91 | 459.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 463.25 | 462.35 | 460.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 461.65 | 461.46 | 460.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 462.20 | 462.20 | 460.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 474.15 | 471.53 | 467.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 475.10 | 472.22 | 468.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 475.00 | 472.22 | 468.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 479.60 | 473.52 | 470.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 475.10 | 474.17 | 471.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 474.30 | 476.08 | 474.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 474.30 | 476.08 | 474.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 479.30 | 476.72 | 474.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 480.10 | 477.08 | 475.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 481.20 | 477.46 | 475.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 479.50 | 477.84 | 475.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 476.00 | 476.39 | 476.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 476.00 | 476.39 | 476.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 471.10 | 475.34 | 475.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 481.20 | 470.47 | 471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 476.50 | 471.68 | 472.04 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 480.75 | 473.49 | 472.83 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 471.80 | 472.80 | 472.89 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 475.50 | 473.34 | 473.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 482.00 | 475.07 | 473.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 496.55 | 497.26 | 491.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 496.55 | 497.26 | 491.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 496.50 | 496.52 | 491.82 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 490.00 | 492.97 | 493.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 489.25 | 491.40 | 492.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 493.30 | 489.14 | 490.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 487.05 | 488.72 | 490.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 486.50 | 488.72 | 490.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 462.17 | 480.16 | 484.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 469.50 | 467.63 | 472.74 | SL hit (close>ema200) qty=0.50 sl=467.63 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 450.40 | 444.88 | 444.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 456.00 | 450.74 | 448.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 504.45 | 509.11 | 494.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 507.55 | 509.11 | 494.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 496.00 | 503.21 | 498.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 495.10 | 503.21 | 498.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 495.60 | 501.69 | 498.22 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 496.20 | 496.86 | 496.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 497.10 | 496.91 | 496.90 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 493.40 | 498.19 | 498.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 492.85 | 496.13 | 497.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 494.45 | 490.71 | 493.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 491.15 | 490.80 | 492.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 496.65 | 490.80 | 492.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 505.80 | 493.80 | 494.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 505.80 | 493.80 | 494.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 508.00 | 496.64 | 495.35 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 500.35 | 504.21 | 504.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 497.40 | 502.31 | 503.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 506.15 | 503.07 | 503.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 507.80 | 504.02 | 504.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 510.55 | 504.02 | 504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 507.55 | 504.73 | 504.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 509.40 | 505.81 | 504.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 514.25 | 519.70 | 514.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 512.00 | 518.16 | 514.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 516.25 | 518.16 | 514.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 515.75 | 516.10 | 513.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 515.30 | 515.43 | 513.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 521.95 | 514.53 | 513.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 515.85 | 514.95 | 514.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 514.75 | 514.95 | 514.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 514.30 | 515.18 | 514.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 514.30 | 515.18 | 514.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 517.50 | 515.64 | 514.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 518.30 | 515.64 | 514.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 515.60 | 516.00 | 515.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 514.65 | 516.00 | 515.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 513.85 | 515.57 | 515.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 513.20 | 515.57 | 515.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 514.35 | 515.33 | 514.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 514.85 | 515.33 | 514.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 513.85 | 515.03 | 514.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 513.30 | 515.03 | 514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 512.85 | 514.60 | 514.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 512.85 | 514.60 | 514.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 508.20 | 513.18 | 514.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 514.20 | 513.44 | 514.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 512.30 | 513.21 | 513.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 12:45:00 | 511.50 | 512.69 | 513.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 511.40 | 512.27 | 513.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 510.50 | 511.12 | 512.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:00:00 | 511.50 | 511.30 | 512.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 511.00 | 511.24 | 512.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 509.25 | 511.24 | 512.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 513.75 | 511.74 | 512.25 | SL hit (close>static) qty=1.00 sl=512.35 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 515.40 | 512.26 | 512.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 517.80 | 513.37 | 512.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 513.20 | 514.19 | 513.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 513.00 | 513.95 | 513.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 514.50 | 513.78 | 513.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 508.75 | 512.54 | 512.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 508.75 | 512.54 | 512.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 503.25 | 510.68 | 511.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 485.75 | 485.28 | 491.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 485.75 | 485.28 | 491.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 484.30 | 485.49 | 490.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:00:00 | 482.85 | 484.96 | 489.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 458.71 | 467.46 | 472.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 467.55 | 467.20 | 471.75 | SL hit (close>ema200) qty=0.50 sl=467.20 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 462.65 | 453.93 | 453.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 475.80 | 460.27 | 456.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 15:15:00 | 471.80 | 472.01 | 466.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:15:00 | 474.50 | 472.01 | 466.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 467.20 | 469.86 | 467.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 467.20 | 469.86 | 467.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 467.45 | 469.38 | 467.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 467.05 | 469.38 | 467.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 466.00 | 468.70 | 467.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 472.00 | 468.70 | 467.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 470.95 | 475.68 | 476.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 470.95 | 475.68 | 476.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 467.95 | 472.67 | 474.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 455.40 | 450.65 | 454.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 454.15 | 451.48 | 453.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 454.15 | 451.48 | 453.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 452.20 | 451.62 | 453.51 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 462.50 | 455.56 | 454.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 466.50 | 458.99 | 456.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 462.50 | 463.52 | 461.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 462.50 | 463.52 | 461.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 461.20 | 463.06 | 461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 461.95 | 463.06 | 461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 461.10 | 462.67 | 461.45 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 457.90 | 460.56 | 460.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 454.85 | 458.78 | 459.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 457.25 | 456.89 | 458.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 458.05 | 456.89 | 458.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 459.20 | 457.04 | 458.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 459.20 | 457.04 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 458.50 | 457.34 | 458.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 456.10 | 457.52 | 458.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 456.30 | 457.44 | 458.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 455.75 | 456.03 | 457.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 483.50 | 458.84 | 457.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 483.50 | 458.84 | 457.21 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 465.85 | 467.13 | 467.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 463.95 | 466.39 | 466.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 473.00 | 465.98 | 466.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 472.00 | 467.19 | 466.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 472.25 | 468.56 | 467.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 469.30 | 469.88 | 468.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 469.55 | 469.82 | 468.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 468.55 | 469.82 | 468.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 469.75 | 469.80 | 468.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 468.75 | 469.80 | 468.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 469.30 | 470.59 | 469.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 469.30 | 470.59 | 469.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 470.00 | 470.47 | 469.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:30:00 | 468.75 | 470.47 | 469.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 467.45 | 469.87 | 469.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 467.45 | 469.87 | 469.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 467.85 | 469.46 | 469.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 464.00 | 469.46 | 469.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 463.30 | 468.23 | 468.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 461.05 | 465.84 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 461.10 | 456.41 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 471.35 | 459.40 | 460.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 471.35 | 459.40 | 460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 12:15:00 | 487.65 | 465.05 | 462.81 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 461.65 | 463.83 | 463.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 456.05 | 461.90 | 463.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 456.00 | 455.63 | 458.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 456.00 | 455.63 | 458.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 449.15 | 446.26 | 449.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 449.15 | 446.26 | 449.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 446.75 | 446.36 | 448.92 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 451.30 | 449.70 | 449.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 463.30 | 452.91 | 451.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 462.00 | 462.15 | 459.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 462.00 | 462.15 | 459.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 458.80 | 461.53 | 459.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 458.85 | 461.53 | 459.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 458.75 | 460.98 | 459.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 457.80 | 460.98 | 459.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 460.10 | 460.80 | 459.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 458.00 | 460.80 | 459.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 456.35 | 459.91 | 459.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 455.80 | 459.91 | 459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 455.45 | 459.02 | 458.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 455.85 | 459.02 | 458.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 456.05 | 458.43 | 458.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 452.30 | 457.20 | 458.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:30:00 | 457.40 | 454.86 | 456.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 455.00 | 454.89 | 456.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 456.80 | 454.89 | 456.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 453.25 | 454.56 | 455.80 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 465.25 | 457.98 | 457.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 468.00 | 461.44 | 458.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 464.00 | 464.61 | 462.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 469.00 | 464.61 | 462.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:30:00 | 470.70 | 466.61 | 463.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 467.75 | 469.30 | 466.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 467.80 | 469.30 | 466.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 467.50 | 469.43 | 467.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 14:15:00 | 467.50 | 469.43 | 467.75 | SL hit (close<ema400) qty=1.00 sl=467.75 alert=retest1 |

### Cycle 30 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 461.70 | 466.02 | 466.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 459.00 | 464.62 | 465.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 456.50 | 459.18 | 461.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 469.65 | 461.74 | 462.10 | SL hit (close>static) qty=1.00 sl=463.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 467.25 | 462.84 | 462.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 470.45 | 465.79 | 464.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 469.40 | 469.93 | 467.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:00:00 | 469.40 | 469.93 | 467.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 467.25 | 469.40 | 467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 466.70 | 469.40 | 467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 471.55 | 469.83 | 467.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 467.15 | 469.83 | 467.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 468.30 | 469.84 | 468.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 469.45 | 469.84 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 471.50 | 470.17 | 468.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 474.85 | 472.20 | 470.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:00:00 | 474.80 | 472.20 | 470.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 475.55 | 472.61 | 470.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 475.00 | 472.69 | 470.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 486.55 | 485.41 | 482.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 480.00 | 481.73 | 481.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 480.00 | 481.73 | 481.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 12:15:00 | 478.30 | 481.04 | 481.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 455.40 | 454.66 | 458.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 455.40 | 454.66 | 458.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 459.50 | 454.45 | 456.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 460.90 | 454.45 | 456.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 461.00 | 455.76 | 456.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 461.00 | 455.76 | 456.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 461.00 | 457.62 | 457.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 463.35 | 460.23 | 459.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 458.50 | 460.41 | 459.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 457.20 | 459.77 | 459.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 457.20 | 459.77 | 459.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 456.20 | 458.48 | 458.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 454.30 | 457.65 | 458.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 456.50 | 455.71 | 456.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 455.55 | 455.68 | 456.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 455.50 | 455.77 | 456.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 455.25 | 455.99 | 456.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 432.72 | 438.90 | 443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 432.49 | 438.90 | 443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 445.60 | 432.30 | 437.14 | SL hit (close>ema200) qty=0.50 sl=432.30 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 443.35 | 439.59 | 439.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 444.30 | 440.53 | 439.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 444.50 | 445.44 | 443.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 444.50 | 445.44 | 443.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 443.35 | 445.02 | 443.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 443.45 | 445.02 | 443.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 445.45 | 445.10 | 443.41 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 439.95 | 442.69 | 442.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 436.85 | 441.52 | 442.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 444.00 | 436.15 | 437.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 442.75 | 437.47 | 438.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 441.55 | 437.47 | 438.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 445.45 | 439.07 | 438.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 447.65 | 441.83 | 440.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 471.40 | 472.29 | 463.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 470.70 | 472.29 | 463.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 462.75 | 469.73 | 464.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 462.75 | 469.73 | 464.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 461.00 | 467.98 | 464.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 460.40 | 467.98 | 464.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 461.30 | 465.72 | 464.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 455.50 | 465.72 | 464.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 457.45 | 462.62 | 462.86 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 466.30 | 462.56 | 462.20 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 457.60 | 461.67 | 462.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 456.00 | 460.15 | 461.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 467.70 | 460.78 | 461.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 465.30 | 461.68 | 461.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 469.45 | 464.02 | 462.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 471.45 | 471.51 | 468.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:30:00 | 471.10 | 471.51 | 468.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 470.00 | 470.79 | 469.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 470.15 | 470.79 | 469.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 468.60 | 470.35 | 469.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 467.45 | 470.35 | 469.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 466.00 | 469.48 | 468.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 466.00 | 469.48 | 468.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 465.60 | 468.24 | 468.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 464.45 | 466.60 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 461.30 | 463.67 | 465.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 468.15 | 464.00 | 465.00 | SL hit (close>static) qty=1.00 sl=467.05 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 478.40 | 466.88 | 466.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 484.20 | 475.31 | 471.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 480.40 | 481.01 | 476.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 480.80 | 480.80 | 477.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 480.80 | 480.80 | 477.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 490.55 | 481.10 | 479.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 485.45 | 484.42 | 481.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 480.45 | 481.08 | 481.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 480.45 | 481.08 | 481.16 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 488.45 | 481.63 | 481.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 491.20 | 483.54 | 482.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 485.40 | 486.79 | 484.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:15:00 | 489.20 | 486.79 | 484.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 489.30 | 487.15 | 485.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 485.40 | 486.66 | 485.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 485.40 | 486.66 | 485.44 | SL hit (close<ema400) qty=1.00 sl=485.44 alert=retest1 |

### Cycle 46 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 481.75 | 488.78 | 488.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 481.40 | 487.31 | 488.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 484.30 | 483.21 | 485.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 484.30 | 483.21 | 485.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 485.00 | 483.70 | 485.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 485.00 | 483.70 | 485.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 483.55 | 483.67 | 485.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 482.30 | 483.67 | 485.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 483.00 | 482.62 | 484.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 458.19 | 468.27 | 473.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 458.85 | 468.27 | 473.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 471.85 | 468.59 | 472.03 | SL hit (close>ema200) qty=0.50 sl=468.59 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 475.25 | 473.43 | 473.36 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 473.25 | 473.46 | 473.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 471.55 | 472.95 | 473.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 480.40 | 472.77 | 473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 475.55 | 473.33 | 473.29 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 472.30 | 473.12 | 473.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 471.25 | 472.75 | 473.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 451.85 | 448.75 | 454.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 451.85 | 448.75 | 454.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 457.90 | 450.58 | 455.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 467.70 | 450.58 | 455.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 461.20 | 452.70 | 455.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 459.45 | 452.70 | 455.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 459.40 | 456.08 | 456.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 457.70 | 456.40 | 456.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 457.70 | 456.40 | 456.24 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 452.95 | 455.61 | 455.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 450.40 | 454.57 | 455.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 449.60 | 447.12 | 450.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 449.60 | 447.12 | 450.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 451.30 | 448.42 | 450.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 451.30 | 448.42 | 450.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 451.10 | 448.95 | 450.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 453.60 | 448.95 | 450.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 450.70 | 449.30 | 450.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:30:00 | 449.50 | 449.38 | 450.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 449.15 | 448.78 | 449.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 447.20 | 448.41 | 449.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 449.45 | 448.68 | 449.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 455.65 | 450.08 | 449.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 455.65 | 450.08 | 449.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 15:15:00 | 470.50 | 454.16 | 451.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 459.00 | 459.20 | 456.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 459.00 | 459.20 | 456.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 456.60 | 458.85 | 456.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 456.60 | 458.85 | 456.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 453.15 | 457.71 | 456.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 453.20 | 457.71 | 456.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 452.40 | 456.65 | 456.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 452.40 | 456.65 | 456.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 447.75 | 454.87 | 455.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 446.00 | 453.09 | 454.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 454.45 | 451.70 | 453.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 469.50 | 455.26 | 454.78 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 455.95 | 458.98 | 459.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 11:15:00 | 453.15 | 457.81 | 458.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 460.75 | 458.49 | 459.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 457.15 | 458.22 | 458.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 459.40 | 458.22 | 458.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 454.50 | 457.07 | 458.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 452.70 | 457.07 | 458.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:30:00 | 453.50 | 455.86 | 457.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 461.05 | 456.48 | 456.96 | SL hit (close>static) qty=1.00 sl=458.75 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 460.70 | 457.90 | 457.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 461.60 | 458.64 | 457.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 455.00 | 457.43 | 457.55 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 14:15:00 | 461.50 | 457.81 | 457.57 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 455.45 | 457.43 | 457.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 452.75 | 456.18 | 456.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 452.35 | 452.27 | 454.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:45:00 | 452.15 | 452.27 | 454.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 454.60 | 452.94 | 454.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 454.90 | 452.94 | 454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 453.30 | 453.01 | 453.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:30:00 | 456.90 | 453.01 | 453.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 450.15 | 452.44 | 453.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 449.05 | 452.05 | 453.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 459.50 | 453.48 | 453.76 | SL hit (close>static) qty=1.00 sl=453.65 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 456.05 | 454.06 | 453.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 473.00 | 457.96 | 455.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 467.00 | 468.79 | 465.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 467.25 | 469.89 | 466.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 480.15 | 474.36 | 469.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 478.20 | 476.79 | 473.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:45:00 | 477.75 | 475.95 | 473.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 483.30 | 486.45 | 486.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 483.30 | 486.45 | 486.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 480.50 | 485.26 | 485.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 472.10 | 483.26 | 483.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 448.50 | 459.99 | 469.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 448.50 | 446.72 | 456.48 | SL hit (close>ema200) qty=0.50 sl=446.72 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 451.45 | 445.25 | 444.98 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 435.65 | 443.79 | 444.59 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 449.50 | 445.27 | 444.71 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 437.85 | 443.79 | 444.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 436.55 | 442.34 | 443.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 439.65 | 439.33 | 441.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:45:00 | 440.55 | 439.33 | 441.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 437.00 | 438.87 | 440.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 427.20 | 438.87 | 440.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 429.80 | 434.90 | 436.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 446.30 | 439.33 | 438.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 446.30 | 439.33 | 438.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 450.80 | 443.61 | 440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 446.70 | 445.83 | 444.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 432.50 | 448.07 | 448.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 432.50 | 448.07 | 448.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 430.30 | 440.00 | 444.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 440.50 | 440.10 | 443.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 440.50 | 440.10 | 443.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 432.00 | 437.19 | 441.40 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 450.20 | 442.77 | 442.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 461.15 | 446.49 | 444.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 457.50 | 458.16 | 452.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 454.20 | 457.37 | 452.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 454.20 | 457.37 | 452.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 454.40 | 457.37 | 452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 452.50 | 456.39 | 452.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 452.50 | 456.39 | 452.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 451.35 | 455.38 | 452.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 451.15 | 455.38 | 452.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 445.30 | 452.55 | 451.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 445.30 | 452.55 | 451.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 442.00 | 450.44 | 450.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 430.90 | 446.53 | 449.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 441.25 | 439.84 | 444.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 441.25 | 439.84 | 444.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 443.40 | 440.55 | 444.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 462.35 | 440.55 | 444.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 458.60 | 444.16 | 445.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 454.95 | 444.16 | 445.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 457.60 | 446.85 | 446.52 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 432.05 | 445.45 | 446.56 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 451.95 | 445.01 | 444.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 471.40 | 453.80 | 449.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 463.50 | 465.12 | 458.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 463.75 | 465.12 | 458.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 458.00 | 463.70 | 458.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 458.00 | 463.70 | 458.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 459.75 | 462.91 | 458.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:00:00 | 461.20 | 461.28 | 458.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:30:00 | 465.60 | 461.82 | 459.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 10:15:00 | 507.32 | 495.66 | 490.89 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 14:45:00 | 461.95 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2025-05-21 10:00:00 | 463.25 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-05-21 12:45:00 | 461.65 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2025-05-21 13:45:00 | 462.20 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2025-05-23 11:30:00 | 475.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-23 12:00:00 | 475.00 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-05-26 09:15:00 | 479.60 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-05-26 11:00:00 | 475.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-27 14:45:00 | 480.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-05-28 09:15:00 | 481.20 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-28 10:15:00 | 479.50 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-12 11:15:00 | 486.50 | 2025-06-13 09:15:00 | 462.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 11:15:00 | 486.50 | 2025-06-16 14:15:00 | 469.50 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2025-07-16 09:15:00 | 516.25 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-16 10:30:00 | 515.75 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-16 13:00:00 | 515.30 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-17 09:15:00 | 521.95 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-21 12:45:00 | 511.50 | 2025-07-22 14:15:00 | 513.75 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-21 13:30:00 | 511.40 | 2025-07-23 09:15:00 | 513.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-22 09:45:00 | 510.50 | 2025-07-23 14:15:00 | 514.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-22 13:00:00 | 511.50 | 2025-07-23 14:15:00 | 514.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-07-22 14:15:00 | 509.25 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-23 09:15:00 | 510.05 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-23 10:30:00 | 510.20 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-23 11:15:00 | 509.75 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-24 14:30:00 | 514.50 | 2025-07-25 09:15:00 | 508.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-30 11:00:00 | 482.85 | 2025-08-04 09:15:00 | 458.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:00:00 | 482.85 | 2025-08-04 11:15:00 | 467.55 | STOP_HIT | 0.50 | 3.17% |
| BUY | retest2 | 2025-08-18 09:15:00 | 472.00 | 2025-08-22 11:15:00 | 470.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-08 12:15:00 | 456.10 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -6.01% |
| SELL | retest2 | 2025-09-08 13:15:00 | 456.30 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -5.96% |
| SELL | retest2 | 2025-09-08 14:30:00 | 455.75 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest1 | 2025-10-15 09:15:00 | 469.00 | 2025-10-16 14:15:00 | 467.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-15 10:30:00 | 470.70 | 2025-10-16 14:15:00 | 467.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-20 14:15:00 | 456.50 | 2025-10-21 13:15:00 | 469.65 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-10-28 10:30:00 | 474.85 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2025-10-28 11:00:00 | 474.80 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-28 11:30:00 | 475.55 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-10-28 12:45:00 | 475.00 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-11-17 13:15:00 | 455.50 | 2025-11-24 10:15:00 | 432.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 455.25 | 2025-11-24 10:15:00 | 432.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 13:15:00 | 455.50 | 2025-11-25 09:15:00 | 445.60 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-11-18 09:15:00 | 455.25 | 2025-11-25 09:15:00 | 445.60 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-12-19 11:45:00 | 461.30 | 2025-12-19 13:15:00 | 468.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-26 09:15:00 | 490.55 | 2025-12-29 13:15:00 | 480.45 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-26 13:15:00 | 485.45 | 2025-12-29 13:15:00 | 480.45 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2026-01-01 09:15:00 | 489.20 | 2026-01-01 12:15:00 | 485.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2026-01-01 11:00:00 | 489.30 | 2026-01-01 12:15:00 | 485.40 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-02 09:15:00 | 489.55 | 2026-01-06 09:15:00 | 481.75 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-07 13:15:00 | 482.30 | 2026-01-12 11:15:00 | 458.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 483.00 | 2026-01-12 11:15:00 | 458.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 482.30 | 2026-01-12 14:15:00 | 471.85 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2026-01-08 10:00:00 | 483.00 | 2026-01-12 14:15:00 | 471.85 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2026-01-22 10:15:00 | 459.45 | 2026-01-23 10:15:00 | 457.70 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-01-23 10:00:00 | 459.40 | 2026-01-23 10:15:00 | 457.70 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-01-28 12:30:00 | 449.50 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-29 09:45:00 | 449.15 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-29 12:00:00 | 447.20 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-29 13:45:00 | 449.45 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-06 10:15:00 | 452.70 | 2026-02-09 10:15:00 | 461.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-06 12:30:00 | 453.50 | 2026-02-09 10:15:00 | 461.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-16 09:15:00 | 449.05 | 2026-02-16 10:15:00 | 459.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-19 09:30:00 | 480.15 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-02-20 09:15:00 | 478.20 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2026-02-20 11:45:00 | 477.75 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2026-03-02 09:15:00 | 472.10 | 2026-03-04 09:15:00 | 448.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 472.10 | 2026-03-05 09:15:00 | 448.50 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:15:00 | 427.20 | 2026-03-17 13:15:00 | 446.30 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-03-17 09:15:00 | 429.80 | 2026-03-17 13:15:00 | 446.30 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-19 12:45:00 | 446.70 | 2026-03-23 09:15:00 | 432.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-09 14:00:00 | 461.20 | 2026-04-21 10:15:00 | 507.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 14:30:00 | 465.60 | 2026-04-21 10:15:00 | 512.16 | TARGET_HIT | 1.00 | 10.00% |
