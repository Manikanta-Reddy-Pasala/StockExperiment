# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 314.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 90 |
| ALERT2 | 89 |
| ALERT2_SKIP | 49 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 143 |
| PARTIAL | 22 |
| TARGET_HIT | 10 |
| STOP_HIT | 140 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 172 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 106
- **Target hits / Stop hits / Partials:** 10 / 140 / 22
- **Avg / median % per leg:** 0.71% / -0.67%
- **Sum % (uncompounded):** 122.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 12 | 23.5% | 3 | 48 | 0 | -0.38% | -19.6% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.71% | 2.8% |
| BUY @ 3rd Alert (retest2) | 47 | 9 | 19.1% | 3 | 44 | 0 | -0.48% | -22.4% |
| SELL (all) | 121 | 54 | 44.6% | 7 | 92 | 22 | 1.17% | 141.6% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.96% | 2.9% |
| SELL @ 3rd Alert (retest2) | 118 | 51 | 43.2% | 7 | 89 | 22 | 1.18% | 138.8% |
| retest1 (combined) | 7 | 6 | 85.7% | 0 | 7 | 0 | 0.82% | 5.7% |
| retest2 (combined) | 165 | 60 | 36.4% | 10 | 133 | 22 | 0.71% | 116.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 473.30 | 469.96 | 469.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 474.05 | 470.78 | 469.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 469.00 | 470.42 | 469.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 15:15:00 | 469.00 | 470.42 | 469.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 469.00 | 470.42 | 469.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 474.95 | 470.42 | 469.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 466.20 | 470.15 | 470.11 | SL hit (close<static) qty=1.00 sl=468.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 467.30 | 469.58 | 469.85 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 475.00 | 470.54 | 470.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 478.95 | 472.40 | 471.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 482.15 | 482.32 | 477.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 482.15 | 482.32 | 477.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 482.15 | 482.32 | 477.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 483.40 | 482.32 | 477.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 485.00 | 483.47 | 480.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 482.90 | 483.47 | 480.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 482.90 | 483.66 | 481.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 483.00 | 483.66 | 481.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 481.75 | 483.20 | 481.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 481.75 | 483.20 | 481.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 482.00 | 482.96 | 481.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 480.20 | 482.96 | 481.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 484.15 | 483.20 | 481.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:15:00 | 484.90 | 483.20 | 481.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 492.00 | 484.96 | 482.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:15:00 | 495.95 | 489.10 | 485.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:45:00 | 498.00 | 490.76 | 486.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 496.00 | 492.75 | 488.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:00:00 | 497.25 | 493.65 | 489.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 487.45 | 493.61 | 490.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 487.45 | 493.61 | 490.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 487.20 | 492.33 | 490.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 486.25 | 492.33 | 490.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 483.95 | 488.68 | 488.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 483.95 | 488.68 | 488.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 478.50 | 485.72 | 487.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 475.40 | 475.11 | 478.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 14:00:00 | 475.40 | 475.11 | 478.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 482.00 | 472.72 | 474.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:45:00 | 486.00 | 472.72 | 474.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 478.80 | 473.94 | 474.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 478.80 | 473.94 | 474.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 480.00 | 475.99 | 475.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 481.00 | 476.99 | 476.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 475.15 | 476.63 | 475.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 475.15 | 476.63 | 475.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 475.15 | 476.63 | 475.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 475.15 | 476.63 | 475.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 476.00 | 476.50 | 475.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 478.20 | 476.50 | 475.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 480.00 | 477.20 | 476.34 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 465.60 | 474.94 | 475.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 453.00 | 470.56 | 473.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 464.00 | 461.22 | 466.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 464.00 | 461.22 | 466.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 464.00 | 461.22 | 466.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 470.00 | 461.22 | 466.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 468.00 | 462.57 | 466.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 468.00 | 462.57 | 466.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 468.90 | 463.84 | 466.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 468.00 | 463.84 | 466.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 466.00 | 464.27 | 466.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:30:00 | 464.00 | 464.27 | 466.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 471.35 | 465.69 | 466.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 471.35 | 465.69 | 466.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 473.00 | 467.15 | 467.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 481.00 | 467.15 | 467.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 477.10 | 469.14 | 468.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 483.00 | 471.91 | 469.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 509.90 | 511.59 | 503.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:45:00 | 506.25 | 511.59 | 503.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 507.85 | 509.54 | 505.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:30:00 | 508.90 | 509.54 | 505.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 506.15 | 508.86 | 505.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:30:00 | 508.80 | 507.70 | 505.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:30:00 | 508.95 | 508.92 | 506.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:00:00 | 509.50 | 512.37 | 511.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:30:00 | 508.70 | 510.41 | 510.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 12:15:00 | 505.10 | 509.35 | 509.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 505.10 | 509.35 | 509.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 504.50 | 507.87 | 509.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 504.45 | 503.63 | 506.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 12:00:00 | 504.45 | 503.63 | 506.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 496.20 | 499.47 | 503.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 494.55 | 499.47 | 503.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 12:15:00 | 495.15 | 498.22 | 501.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 13:00:00 | 495.10 | 497.60 | 501.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 506.40 | 498.54 | 500.44 | SL hit (close>static) qty=1.00 sl=504.90 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 507.45 | 502.60 | 502.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 523.50 | 506.74 | 504.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 512.00 | 515.55 | 511.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 512.00 | 515.55 | 511.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 512.00 | 515.55 | 511.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 512.00 | 515.55 | 511.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 520.00 | 516.44 | 512.00 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 504.05 | 510.76 | 511.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 503.95 | 509.40 | 510.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 509.00 | 507.57 | 509.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 509.00 | 507.57 | 509.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 509.00 | 507.57 | 509.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 510.00 | 507.57 | 509.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 507.00 | 507.46 | 509.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 505.15 | 507.46 | 509.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:45:00 | 505.25 | 506.96 | 508.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:00:00 | 505.00 | 506.57 | 508.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:00:00 | 504.00 | 505.04 | 507.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 498.70 | 500.99 | 504.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:45:00 | 497.10 | 500.99 | 504.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 504.00 | 501.59 | 503.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 507.00 | 501.59 | 503.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 505.00 | 502.28 | 503.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 503.20 | 502.28 | 503.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 502.00 | 502.22 | 503.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 509.00 | 503.76 | 503.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 509.00 | 503.76 | 503.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 512.00 | 506.81 | 505.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 508.05 | 508.47 | 506.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 508.05 | 508.47 | 506.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 508.00 | 508.37 | 506.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 508.00 | 508.37 | 506.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 507.95 | 508.29 | 506.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 508.95 | 508.29 | 506.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 507.70 | 508.17 | 506.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 509.95 | 508.17 | 506.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 506.25 | 507.79 | 506.79 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 14:15:00 | 503.05 | 506.55 | 506.61 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 12:15:00 | 507.00 | 506.60 | 506.55 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 503.25 | 505.93 | 506.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 10:15:00 | 503.20 | 505.09 | 505.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 13:15:00 | 504.95 | 504.37 | 505.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 13:15:00 | 504.95 | 504.37 | 505.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 504.95 | 504.37 | 505.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 504.95 | 504.37 | 505.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 519.00 | 507.30 | 506.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 528.00 | 518.87 | 514.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 518.95 | 521.12 | 516.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 518.95 | 521.12 | 516.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 518.95 | 521.12 | 516.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 518.95 | 521.12 | 516.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 520.00 | 520.89 | 516.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 514.00 | 520.89 | 516.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 519.00 | 520.21 | 517.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:30:00 | 518.00 | 520.21 | 517.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 517.50 | 519.12 | 517.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:45:00 | 517.05 | 519.12 | 517.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 518.05 | 518.90 | 517.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 524.00 | 518.90 | 517.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 533.00 | 521.72 | 519.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:30:00 | 518.00 | 521.72 | 519.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 545.00 | 555.28 | 549.77 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 13:15:00 | 540.80 | 547.16 | 547.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 539.05 | 544.52 | 545.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 529.35 | 527.60 | 534.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 529.35 | 527.60 | 534.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 527.75 | 527.63 | 533.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 534.90 | 527.63 | 533.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 535.15 | 529.13 | 533.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 542.95 | 529.13 | 533.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 533.20 | 529.95 | 533.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:30:00 | 534.00 | 529.95 | 533.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 527.45 | 529.45 | 533.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 533.90 | 529.45 | 533.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 534.50 | 530.46 | 533.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 534.50 | 530.46 | 533.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 533.00 | 530.97 | 533.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 534.50 | 530.97 | 533.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 527.00 | 530.17 | 532.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 513.55 | 529.54 | 531.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 522.75 | 528.18 | 531.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 523.00 | 527.81 | 530.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:45:00 | 524.55 | 527.05 | 530.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 535.65 | 528.28 | 530.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 535.95 | 528.28 | 530.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 534.20 | 529.46 | 530.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:30:00 | 531.25 | 530.17 | 530.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:45:00 | 531.35 | 530.49 | 530.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 535.60 | 531.51 | 531.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 535.60 | 531.51 | 531.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 11:15:00 | 536.95 | 533.21 | 532.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 12:15:00 | 533.00 | 533.16 | 532.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 533.00 | 533.16 | 532.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 533.00 | 533.16 | 532.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 534.95 | 533.16 | 532.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 531.10 | 532.75 | 532.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 531.10 | 532.75 | 532.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 528.25 | 531.85 | 531.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 528.25 | 531.85 | 531.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 15:15:00 | 527.05 | 530.89 | 531.37 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 536.05 | 531.92 | 531.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 550.40 | 537.48 | 534.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 534.00 | 536.78 | 534.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 15:15:00 | 534.00 | 536.78 | 534.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 534.00 | 536.78 | 534.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 525.00 | 536.78 | 534.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 525.05 | 534.44 | 533.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:30:00 | 523.85 | 534.44 | 533.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 10:15:00 | 525.95 | 532.74 | 533.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 524.50 | 529.84 | 531.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 12:15:00 | 525.00 | 524.81 | 527.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 12:15:00 | 525.00 | 524.81 | 527.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 525.00 | 524.81 | 527.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 525.00 | 524.81 | 527.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 522.70 | 523.04 | 525.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 10:30:00 | 520.40 | 522.56 | 525.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 11:30:00 | 520.35 | 522.21 | 524.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 14:00:00 | 520.40 | 521.72 | 524.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 09:45:00 | 517.70 | 518.97 | 522.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 494.38 | 508.95 | 513.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 494.33 | 508.95 | 513.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 494.38 | 508.95 | 513.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:15:00 | 491.81 | 502.45 | 509.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 500.60 | 492.89 | 499.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-06 11:15:00 | 500.60 | 492.89 | 499.49 | SL hit (close>ema200) qty=0.50 sl=492.89 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 494.50 | 486.64 | 486.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 500.30 | 490.69 | 488.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 494.10 | 495.45 | 492.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:45:00 | 495.70 | 495.45 | 492.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 501.55 | 501.49 | 498.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 498.65 | 501.49 | 498.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 501.00 | 501.66 | 499.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 501.00 | 501.66 | 499.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 501.80 | 501.34 | 499.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:15:00 | 497.85 | 501.34 | 499.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 497.85 | 500.64 | 499.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 508.50 | 500.64 | 499.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 500.85 | 505.75 | 506.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 500.85 | 505.75 | 506.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 497.30 | 504.06 | 505.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 494.00 | 493.67 | 496.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:30:00 | 494.00 | 493.67 | 496.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 494.75 | 493.44 | 495.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 495.90 | 493.44 | 495.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 486.60 | 492.07 | 494.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 484.65 | 488.64 | 491.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 484.90 | 487.74 | 490.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:45:00 | 484.85 | 486.63 | 489.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 483.95 | 484.60 | 487.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 477.45 | 478.95 | 482.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 475.00 | 478.95 | 482.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:45:00 | 474.45 | 476.17 | 477.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:15:00 | 475.00 | 476.04 | 477.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:00:00 | 475.00 | 475.83 | 477.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 471.80 | 474.95 | 476.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 462.80 | 473.13 | 474.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 460.42 | 470.33 | 473.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 460.65 | 470.33 | 473.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 460.61 | 470.33 | 473.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 459.75 | 470.33 | 473.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 15:15:00 | 465.00 | 463.76 | 467.95 | SL hit (close>ema200) qty=0.50 sl=463.76 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 481.95 | 471.03 | 470.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 494.00 | 477.66 | 473.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 482.85 | 484.18 | 479.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 482.85 | 484.18 | 479.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 481.25 | 483.26 | 480.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 481.05 | 483.26 | 480.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 491.55 | 484.92 | 481.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 494.80 | 484.92 | 481.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:45:00 | 493.95 | 487.00 | 482.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 492.90 | 490.19 | 485.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 13:15:00 | 483.00 | 486.11 | 486.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 483.00 | 486.11 | 486.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 481.40 | 485.17 | 485.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 485.40 | 483.25 | 484.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 11:15:00 | 485.40 | 483.25 | 484.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 485.40 | 483.25 | 484.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 487.00 | 483.25 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 484.50 | 483.50 | 484.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:30:00 | 485.70 | 483.50 | 484.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 483.55 | 483.51 | 484.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:45:00 | 486.00 | 483.51 | 484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 493.10 | 485.43 | 485.15 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 480.60 | 484.61 | 485.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 14:15:00 | 479.95 | 483.37 | 484.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 478.25 | 478.04 | 480.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 478.25 | 478.04 | 480.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 480.90 | 478.61 | 480.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 476.85 | 478.61 | 480.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 476.70 | 478.23 | 480.37 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 482.50 | 480.09 | 479.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 489.50 | 481.97 | 480.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 483.60 | 484.95 | 482.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 483.60 | 484.95 | 482.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 483.60 | 484.95 | 482.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:45:00 | 483.00 | 484.95 | 482.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 483.10 | 484.20 | 482.98 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 478.50 | 481.86 | 482.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 477.80 | 481.05 | 481.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 478.80 | 476.69 | 478.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 478.80 | 476.69 | 478.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 478.80 | 476.69 | 478.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:45:00 | 474.75 | 476.32 | 477.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:30:00 | 474.45 | 475.15 | 476.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 473.15 | 475.60 | 476.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 451.01 | 457.99 | 461.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 459.60 | 457.99 | 461.67 | SL hit (close>static) qty=0.50 sl=457.99 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 472.60 | 463.33 | 462.99 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 15:15:00 | 461.15 | 466.71 | 467.11 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 11:15:00 | 468.80 | 465.07 | 464.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 12:15:00 | 470.00 | 466.06 | 465.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 481.20 | 485.27 | 481.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 481.20 | 485.27 | 481.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 481.20 | 485.27 | 481.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 480.25 | 485.27 | 481.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 482.00 | 484.61 | 481.63 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 476.35 | 480.17 | 480.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 465.50 | 477.23 | 478.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 11:15:00 | 448.75 | 446.44 | 450.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 11:15:00 | 448.75 | 446.44 | 450.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 448.75 | 446.44 | 450.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:30:00 | 450.10 | 446.44 | 450.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 451.60 | 446.71 | 448.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 451.60 | 446.71 | 448.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 445.25 | 446.42 | 448.45 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 456.20 | 450.32 | 449.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 457.00 | 452.60 | 450.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 452.95 | 453.06 | 451.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 452.95 | 453.06 | 451.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 458.90 | 454.70 | 452.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:00:00 | 459.50 | 455.66 | 453.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 459.15 | 456.39 | 453.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 459.00 | 456.91 | 454.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 11:15:00 | 452.65 | 456.07 | 455.28 | SL hit (close<static) qty=1.00 sl=452.75 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 466.40 | 477.41 | 478.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 463.55 | 468.41 | 472.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 457.70 | 456.95 | 461.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 10:45:00 | 453.00 | 456.38 | 460.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 12:30:00 | 453.30 | 455.50 | 459.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 445.60 | 442.27 | 447.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 445.60 | 442.27 | 447.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 446.30 | 444.06 | 446.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:15:00 | 447.50 | 444.06 | 446.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 447.50 | 444.75 | 446.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-22 15:15:00 | 447.50 | 444.75 | 446.70 | SL hit (close>ema400) qty=1.00 sl=446.70 alert=retest1 |

### Cycle 35 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 449.65 | 446.70 | 446.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 10:15:00 | 453.90 | 448.14 | 447.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 452.05 | 452.36 | 450.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 12:15:00 | 450.65 | 452.01 | 450.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 450.65 | 452.01 | 450.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 451.45 | 452.01 | 450.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 451.40 | 451.89 | 450.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 453.05 | 451.33 | 450.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 450.05 | 451.07 | 450.51 | SL hit (close<static) qty=1.00 sl=450.15 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 13:15:00 | 447.70 | 449.88 | 450.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 446.95 | 448.82 | 449.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 450.00 | 449.05 | 449.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 10:15:00 | 450.00 | 449.05 | 449.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 450.00 | 449.05 | 449.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 449.60 | 449.05 | 449.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 450.90 | 449.42 | 449.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 451.20 | 449.42 | 449.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 453.55 | 450.25 | 450.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 454.95 | 451.79 | 450.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 466.70 | 467.37 | 461.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 466.70 | 467.37 | 461.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 476.75 | 478.48 | 474.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:45:00 | 474.05 | 478.48 | 474.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 471.30 | 476.49 | 474.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:45:00 | 473.90 | 476.49 | 474.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 471.95 | 475.58 | 474.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 471.15 | 475.58 | 474.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 467.00 | 473.36 | 473.49 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 482.10 | 475.11 | 474.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 11:15:00 | 485.60 | 477.20 | 475.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 12:15:00 | 481.30 | 483.08 | 480.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:30:00 | 481.50 | 483.08 | 480.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 479.15 | 482.29 | 480.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 479.15 | 482.29 | 480.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 493.25 | 484.49 | 481.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 502.05 | 489.90 | 484.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 10:45:00 | 497.00 | 505.07 | 502.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 501.70 | 503.19 | 503.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 501.70 | 503.19 | 503.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 499.00 | 501.93 | 502.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 489.50 | 486.94 | 490.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 13:15:00 | 489.50 | 486.94 | 490.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 489.50 | 486.94 | 490.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 494.05 | 486.94 | 490.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 485.85 | 486.34 | 489.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 492.00 | 486.34 | 489.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 487.05 | 486.48 | 489.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 484.45 | 486.08 | 488.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:00:00 | 480.50 | 472.94 | 473.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 484.25 | 475.20 | 474.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 484.25 | 475.20 | 474.62 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 471.05 | 475.81 | 476.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 470.00 | 474.65 | 475.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 477.05 | 472.74 | 474.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 477.05 | 472.74 | 474.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 477.05 | 472.74 | 474.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 477.05 | 472.74 | 474.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 484.45 | 475.08 | 475.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 487.20 | 475.08 | 475.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 488.80 | 477.82 | 476.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 491.60 | 480.58 | 477.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 495.90 | 499.37 | 495.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 495.90 | 499.37 | 495.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 495.90 | 499.37 | 495.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 499.35 | 499.37 | 495.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 485.70 | 496.63 | 494.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 485.70 | 496.63 | 494.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 483.50 | 494.01 | 493.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 482.60 | 494.01 | 493.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 490.00 | 493.21 | 493.22 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 495.30 | 492.89 | 492.61 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 489.70 | 492.25 | 492.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 485.90 | 491.02 | 491.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 446.05 | 444.73 | 450.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 446.70 | 444.73 | 450.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 446.20 | 445.02 | 450.54 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 456.30 | 452.51 | 452.05 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 450.70 | 452.62 | 452.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 450.00 | 452.10 | 452.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 451.80 | 451.29 | 451.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 451.80 | 451.29 | 451.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 451.80 | 451.29 | 451.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 451.50 | 451.29 | 451.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 453.00 | 451.63 | 452.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 453.00 | 451.63 | 452.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 452.70 | 451.84 | 452.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:45:00 | 454.95 | 451.84 | 452.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 455.35 | 452.89 | 452.57 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 446.55 | 452.24 | 452.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 446.05 | 451.00 | 452.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 452.40 | 450.55 | 451.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 452.40 | 450.55 | 451.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 452.40 | 450.55 | 451.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 452.40 | 450.55 | 451.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 452.80 | 451.00 | 451.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 450.95 | 451.00 | 451.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 456.55 | 452.11 | 452.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 455.90 | 452.11 | 452.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 455.90 | 452.87 | 452.68 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 14:15:00 | 451.20 | 452.41 | 452.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 15:15:00 | 449.05 | 451.74 | 452.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 418.50 | 414.64 | 424.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 418.50 | 414.64 | 424.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 423.40 | 417.43 | 422.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 423.40 | 417.43 | 422.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 423.45 | 418.63 | 422.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 423.45 | 418.63 | 422.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 424.05 | 419.72 | 422.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 423.60 | 419.72 | 422.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 425.45 | 420.86 | 422.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 423.80 | 420.86 | 422.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 430.60 | 424.99 | 424.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 437.00 | 427.39 | 425.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 429.30 | 430.09 | 427.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 429.30 | 430.09 | 427.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 427.25 | 429.46 | 427.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 430.40 | 429.65 | 427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 433.50 | 436.27 | 434.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:15:00 | 431.85 | 436.27 | 434.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 434.20 | 435.86 | 434.08 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 429.95 | 432.55 | 432.90 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 443.25 | 434.60 | 433.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 10:15:00 | 444.15 | 436.51 | 434.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 458.05 | 458.82 | 453.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:45:00 | 458.50 | 458.82 | 453.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 459.05 | 458.94 | 455.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 460.80 | 458.94 | 455.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 14:15:00 | 447.85 | 455.26 | 454.69 | SL hit (close<static) qty=1.00 sl=452.55 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 446.00 | 453.41 | 453.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 429.10 | 448.04 | 451.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 430.00 | 428.30 | 437.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 409.30 | 428.30 | 437.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 417.85 | 415.63 | 422.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 421.10 | 415.63 | 422.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 414.70 | 415.40 | 420.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:45:00 | 419.75 | 415.40 | 420.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 418.75 | 416.14 | 420.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 413.75 | 415.61 | 419.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 407.75 | 400.10 | 403.29 | SL hit (close>ema400) qty=1.00 sl=403.29 alert=retest1 |

### Cycle 57 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 408.85 | 403.53 | 403.52 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 402.10 | 404.69 | 404.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 396.60 | 403.08 | 404.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 10:15:00 | 386.70 | 383.32 | 387.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 10:15:00 | 386.70 | 383.32 | 387.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 386.70 | 383.32 | 387.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 386.70 | 383.32 | 387.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 380.00 | 382.65 | 386.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:45:00 | 378.95 | 382.15 | 386.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 378.25 | 381.47 | 385.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 373.85 | 378.89 | 383.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 11:15:00 | 360.00 | 374.05 | 380.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 370.15 | 368.80 | 374.92 | SL hit (close>ema200) qty=0.50 sl=368.80 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 385.00 | 375.91 | 375.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 391.55 | 384.14 | 380.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 385.00 | 385.49 | 381.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 385.00 | 385.49 | 381.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 382.50 | 384.59 | 382.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 382.50 | 384.59 | 382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 382.95 | 384.26 | 382.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 389.10 | 384.26 | 382.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 388.50 | 385.11 | 382.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 390.35 | 385.11 | 382.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 375.80 | 383.87 | 383.62 | SL hit (close<static) qty=1.00 sl=380.90 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 375.35 | 382.16 | 382.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 367.85 | 379.30 | 381.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 352.20 | 351.26 | 355.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 352.20 | 351.26 | 355.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 352.20 | 351.26 | 355.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 348.35 | 351.13 | 353.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 348.40 | 350.59 | 353.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 362.40 | 352.25 | 353.62 | SL hit (close>static) qty=1.00 sl=355.95 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 358.40 | 355.10 | 354.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 360.35 | 356.15 | 355.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 371.40 | 371.74 | 367.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 375.95 | 371.74 | 367.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 375.00 | 378.67 | 376.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-24 15:15:00 | 375.00 | 378.67 | 376.54 | SL hit (close<ema400) qty=1.00 sl=376.54 alert=retest1 |

### Cycle 62 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 368.85 | 375.33 | 375.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 13:15:00 | 364.00 | 371.87 | 373.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 369.45 | 368.52 | 371.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 369.45 | 368.52 | 371.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 369.45 | 368.52 | 371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 369.45 | 368.52 | 371.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 361.25 | 362.60 | 365.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 363.40 | 362.60 | 365.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 365.15 | 363.11 | 365.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 365.20 | 363.11 | 365.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 362.25 | 362.93 | 365.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 365.95 | 362.93 | 365.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 365.50 | 363.45 | 365.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:00:00 | 361.25 | 363.26 | 364.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 371.20 | 366.48 | 365.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 371.20 | 366.48 | 365.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 10:15:00 | 372.15 | 367.61 | 366.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 14:15:00 | 370.35 | 370.45 | 368.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 15:00:00 | 370.35 | 370.45 | 368.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 368.70 | 369.78 | 368.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 374.10 | 371.13 | 369.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 373.65 | 371.13 | 369.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 374.15 | 371.43 | 370.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 375.90 | 371.75 | 370.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 375.15 | 372.53 | 371.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 363.00 | 371.18 | 370.86 | SL hit (close<static) qty=1.00 sl=364.45 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 362.50 | 369.45 | 370.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 360.25 | 367.61 | 369.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 365.95 | 365.80 | 367.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 15:15:00 | 365.95 | 365.80 | 367.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 365.95 | 365.80 | 367.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 347.75 | 365.80 | 367.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 15:15:00 | 363.50 | 359.07 | 358.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 363.50 | 359.07 | 358.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 370.35 | 361.33 | 360.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 382.85 | 383.86 | 378.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:00:00 | 382.85 | 383.86 | 378.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 392.90 | 395.00 | 392.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 392.90 | 395.00 | 392.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 392.50 | 394.50 | 392.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 402.65 | 394.50 | 392.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 402.85 | 409.07 | 409.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 402.85 | 409.07 | 409.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 401.90 | 407.63 | 408.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 408.80 | 407.15 | 408.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 408.80 | 407.15 | 408.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 408.80 | 407.15 | 408.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 409.20 | 407.15 | 408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 406.10 | 406.94 | 407.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:45:00 | 405.80 | 407.00 | 407.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 406.00 | 407.00 | 407.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 15:15:00 | 409.80 | 406.99 | 407.71 | SL hit (close>static) qty=1.00 sl=409.20 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 15:15:00 | 405.00 | 402.19 | 402.00 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 400.50 | 401.78 | 401.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 14:15:00 | 398.50 | 400.84 | 401.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 15:15:00 | 401.55 | 400.98 | 401.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 15:15:00 | 401.55 | 400.98 | 401.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 401.55 | 400.98 | 401.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 393.00 | 400.98 | 401.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 410.75 | 396.67 | 395.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 410.75 | 396.67 | 395.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 411.30 | 399.60 | 397.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 400.20 | 402.69 | 399.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 400.20 | 402.69 | 399.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 401.00 | 402.35 | 399.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 400.00 | 402.35 | 399.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 401.85 | 402.25 | 399.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:30:00 | 404.30 | 403.48 | 400.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 420.00 | 421.14 | 421.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 420.00 | 421.14 | 421.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 417.45 | 419.87 | 420.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 410.00 | 409.86 | 413.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 15:15:00 | 409.00 | 408.70 | 410.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 409.00 | 408.70 | 410.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 414.95 | 409.41 | 411.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 411.40 | 409.81 | 411.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:15:00 | 414.55 | 409.81 | 411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 417.30 | 411.31 | 411.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 417.30 | 411.31 | 411.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 416.35 | 412.32 | 412.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 418.85 | 413.62 | 412.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 418.00 | 418.12 | 416.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 414.50 | 418.12 | 416.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 425.85 | 419.66 | 417.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 414.15 | 419.66 | 417.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 418.45 | 419.69 | 418.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 418.45 | 419.69 | 418.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 423.85 | 420.52 | 418.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 417.30 | 420.52 | 418.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 421.50 | 420.67 | 419.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 422.45 | 420.63 | 419.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 417.80 | 419.89 | 419.42 | SL hit (close<static) qty=1.00 sl=419.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 415.20 | 419.19 | 419.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 415.05 | 418.37 | 418.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 416.50 | 416.13 | 417.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 416.50 | 416.13 | 417.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 416.20 | 416.14 | 416.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 415.70 | 416.14 | 416.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 416.45 | 416.20 | 416.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 416.45 | 416.20 | 416.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 416.50 | 416.26 | 416.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 416.15 | 416.26 | 416.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 416.05 | 416.22 | 416.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 413.35 | 416.18 | 416.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 414.30 | 415.80 | 416.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 414.45 | 415.76 | 416.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 415.00 | 412.15 | 411.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 415.00 | 412.15 | 411.90 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 412.20 | 412.69 | 412.75 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 421.15 | 414.38 | 413.51 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 410.00 | 413.95 | 414.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 409.25 | 413.01 | 413.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 397.80 | 397.54 | 401.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 397.80 | 397.54 | 401.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 401.55 | 398.34 | 401.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 399.80 | 398.34 | 401.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 399.00 | 398.47 | 401.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 418.40 | 398.47 | 401.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 415.50 | 401.88 | 402.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 427.85 | 401.88 | 402.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 418.40 | 405.18 | 404.11 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 396.90 | 403.92 | 404.82 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 407.60 | 403.09 | 403.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 408.50 | 405.96 | 404.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 405.35 | 406.30 | 405.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 405.35 | 406.30 | 405.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 406.85 | 406.41 | 405.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 405.95 | 406.41 | 405.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 405.95 | 406.32 | 405.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 408.85 | 406.32 | 405.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:00:00 | 408.50 | 407.29 | 405.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-03 14:15:00 | 449.74 | 436.26 | 426.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 439.45 | 441.64 | 441.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 436.80 | 440.04 | 440.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 429.40 | 429.30 | 432.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 428.70 | 429.30 | 432.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 434.60 | 430.36 | 432.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 434.60 | 430.36 | 432.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 435.95 | 431.48 | 432.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 436.70 | 431.48 | 432.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 432.70 | 432.61 | 433.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 430.80 | 432.51 | 433.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 435.50 | 432.51 | 432.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 435.50 | 432.51 | 432.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 13:15:00 | 439.80 | 435.54 | 434.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 436.60 | 437.46 | 435.97 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 422.65 | 433.93 | 434.59 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 441.70 | 435.76 | 435.25 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 430.85 | 439.83 | 440.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 422.65 | 431.56 | 436.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 411.15 | 409.45 | 414.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 411.15 | 409.45 | 414.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 418.15 | 411.19 | 415.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 417.40 | 411.19 | 415.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 416.95 | 412.34 | 415.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 414.40 | 412.44 | 415.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 393.68 | 397.01 | 399.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 394.95 | 391.97 | 394.79 | SL hit (close>ema200) qty=0.50 sl=391.97 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 397.25 | 396.17 | 396.04 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 394.55 | 395.99 | 396.01 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 400.40 | 396.87 | 396.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 10:15:00 | 402.95 | 398.80 | 397.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 397.00 | 398.50 | 397.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 396.65 | 398.50 | 397.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 397.50 | 398.30 | 397.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 397.10 | 398.30 | 397.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 402.50 | 398.80 | 397.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 403.70 | 399.23 | 398.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 404.00 | 400.11 | 399.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 421.30 | 425.89 | 426.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 421.30 | 425.89 | 426.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 417.90 | 423.32 | 424.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 416.85 | 414.83 | 417.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 416.85 | 414.83 | 417.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 417.55 | 415.37 | 417.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 417.10 | 415.37 | 417.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 416.80 | 415.66 | 417.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 415.30 | 415.75 | 417.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 423.90 | 417.55 | 417.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 423.90 | 417.55 | 417.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 425.20 | 419.08 | 418.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 426.70 | 427.78 | 424.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 426.70 | 427.78 | 424.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 428.05 | 428.35 | 425.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 425.30 | 428.35 | 425.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 427.00 | 427.55 | 425.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 425.50 | 427.55 | 425.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 426.55 | 427.86 | 426.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 433.50 | 427.47 | 426.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 432.30 | 428.28 | 426.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 432.45 | 428.28 | 426.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 424.00 | 426.85 | 426.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 424.00 | 426.85 | 426.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 422.50 | 425.41 | 426.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 425.90 | 424.65 | 425.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 426.00 | 424.65 | 425.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 426.00 | 424.92 | 425.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 425.25 | 424.90 | 425.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 425.20 | 425.37 | 425.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 424.10 | 425.37 | 425.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 425.15 | 425.21 | 425.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 424.15 | 425.00 | 425.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 422.90 | 424.48 | 425.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 422.80 | 418.02 | 418.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 423.75 | 419.05 | 418.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 423.75 | 419.05 | 418.83 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 416.10 | 421.53 | 421.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 415.00 | 420.23 | 421.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 410.20 | 407.17 | 409.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 412.70 | 407.17 | 409.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 410.60 | 407.86 | 409.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 411.50 | 407.86 | 409.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 408.45 | 407.98 | 409.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 402.55 | 408.05 | 409.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 406.05 | 407.65 | 408.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 407.05 | 407.23 | 408.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 412.60 | 405.02 | 405.75 | SL hit (close>static) qty=1.00 sl=410.60 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 410.20 | 406.96 | 406.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 411.30 | 408.60 | 407.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 409.40 | 409.40 | 408.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 11:15:00 | 414.50 | 410.44 | 408.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 12:45:00 | 414.90 | 411.38 | 409.65 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 14:15:00 | 414.60 | 411.80 | 410.00 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 422.10 | 423.39 | 420.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 419.50 | 423.39 | 420.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 422.50 | 423.39 | 420.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 418.95 | 422.50 | 420.65 | SL hit (close<ema400) qty=1.00 sl=420.65 alert=retest1 |

### Cycle 94 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 428.80 | 432.05 | 432.16 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 435.05 | 432.01 | 432.00 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 15:15:00 | 428.75 | 432.00 | 432.19 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 434.95 | 432.59 | 432.44 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 427.10 | 432.67 | 433.06 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 434.65 | 431.12 | 431.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 438.35 | 433.03 | 431.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 433.30 | 434.39 | 433.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 433.30 | 434.39 | 433.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 434.95 | 434.50 | 433.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 436.35 | 434.50 | 433.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 435.25 | 434.65 | 433.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 432.25 | 434.17 | 433.35 | SL hit (close<static) qty=1.00 sl=432.70 alert=retest2 |

### Cycle 100 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 428.35 | 432.09 | 432.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 418.65 | 428.43 | 430.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 425.60 | 424.75 | 427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 426.80 | 424.75 | 427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 425.40 | 425.08 | 427.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 420.00 | 424.35 | 426.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 419.95 | 424.35 | 426.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 419.25 | 417.66 | 420.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 420.25 | 418.18 | 420.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 419.45 | 418.43 | 420.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 420.90 | 418.43 | 420.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 418.10 | 418.54 | 420.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 422.75 | 420.82 | 420.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 422.75 | 420.82 | 420.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 437.00 | 424.72 | 422.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 467.15 | 467.30 | 455.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:30:00 | 465.80 | 467.30 | 455.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 449.10 | 463.38 | 461.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 454.75 | 463.38 | 461.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 447.80 | 460.26 | 460.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 448.00 | 460.26 | 460.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 446.00 | 457.41 | 458.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 13:15:00 | 438.75 | 451.50 | 455.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 449.90 | 445.85 | 451.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 449.90 | 445.85 | 451.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 447.00 | 446.08 | 450.73 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 468.30 | 456.24 | 454.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 474.75 | 459.94 | 456.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 478.25 | 479.25 | 471.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 478.25 | 479.25 | 471.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 480.30 | 481.33 | 478.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 480.30 | 481.33 | 478.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 484.10 | 485.05 | 482.62 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 467.00 | 479.54 | 480.70 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 496.35 | 481.60 | 479.80 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 485.60 | 486.17 | 486.17 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 491.60 | 487.25 | 486.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 504.20 | 491.36 | 489.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 498.00 | 499.60 | 495.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:00:00 | 498.00 | 499.60 | 495.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 493.25 | 498.33 | 495.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 493.45 | 498.33 | 495.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 491.45 | 496.96 | 495.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 491.45 | 496.96 | 495.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 496.00 | 496.76 | 495.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 496.55 | 495.17 | 494.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 490.80 | 494.30 | 494.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 490.80 | 494.30 | 494.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 488.30 | 493.10 | 493.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 497.20 | 491.05 | 492.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 497.20 | 491.05 | 492.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 501.30 | 493.10 | 493.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 501.30 | 493.10 | 493.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 498.80 | 494.24 | 493.70 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 487.80 | 493.30 | 493.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 482.00 | 489.45 | 491.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 461.25 | 460.53 | 467.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:30:00 | 461.20 | 460.53 | 467.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 477.85 | 459.86 | 461.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 477.85 | 459.86 | 461.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 485.00 | 464.89 | 463.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 491.25 | 477.95 | 471.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 490.95 | 494.03 | 485.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 484.45 | 492.11 | 485.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 484.45 | 492.11 | 485.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 484.45 | 492.11 | 485.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 482.95 | 490.28 | 485.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 481.75 | 490.28 | 485.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 480.10 | 488.24 | 484.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 480.10 | 488.24 | 484.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 475.70 | 485.73 | 483.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 475.70 | 485.73 | 483.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 475.10 | 482.23 | 482.59 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 489.35 | 483.35 | 483.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 493.25 | 487.32 | 485.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 491.00 | 492.77 | 489.76 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 486.15 | 489.28 | 489.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 483.30 | 487.11 | 487.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 484.50 | 483.77 | 485.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 482.15 | 483.19 | 484.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 486.00 | 484.30 | 484.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 486.00 | 484.30 | 484.16 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 477.50 | 482.88 | 483.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 475.50 | 481.40 | 482.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 12:15:00 | 471.95 | 470.66 | 474.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 471.95 | 470.66 | 474.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 459.80 | 462.97 | 467.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 458.50 | 462.03 | 466.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 458.25 | 462.03 | 466.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 458.10 | 461.53 | 465.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 457.25 | 461.02 | 464.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 458.70 | 459.15 | 461.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 458.70 | 459.15 | 461.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 460.50 | 459.42 | 461.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 457.45 | 459.01 | 460.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 461.80 | 459.36 | 459.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 461.80 | 459.36 | 459.35 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 457.90 | 459.60 | 459.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 456.70 | 459.02 | 459.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 460.65 | 456.78 | 457.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 462.90 | 456.78 | 457.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 459.80 | 457.38 | 457.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 463.70 | 457.38 | 457.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 466.00 | 459.11 | 458.58 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 452.15 | 458.82 | 459.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 447.90 | 456.64 | 458.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 441.00 | 440.13 | 445.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 442.60 | 440.13 | 445.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 440.60 | 440.23 | 444.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 438.05 | 439.78 | 443.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 437.75 | 438.75 | 442.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 437.65 | 438.81 | 441.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 437.20 | 438.59 | 441.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 438.40 | 438.55 | 441.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 438.40 | 438.55 | 441.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 434.80 | 434.30 | 437.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 436.95 | 434.30 | 437.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 434.55 | 433.38 | 435.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 434.55 | 433.38 | 435.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 416.15 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 415.86 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 415.77 | 422.33 | 427.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 415.34 | 420.77 | 426.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 394.25 | 406.48 | 416.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 402.30 | 398.97 | 398.81 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 394.05 | 397.85 | 398.36 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 403.25 | 399.20 | 398.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 12:15:00 | 405.25 | 401.11 | 399.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 453.75 | 461.58 | 448.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 453.75 | 461.58 | 448.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 446.10 | 458.49 | 448.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 446.10 | 458.49 | 448.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 440.10 | 454.81 | 447.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 440.10 | 454.81 | 447.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 435.80 | 443.00 | 443.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 432.45 | 439.54 | 441.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 394.55 | 393.52 | 400.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 394.55 | 393.52 | 400.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 397.30 | 390.07 | 393.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 397.30 | 390.07 | 393.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 396.15 | 391.28 | 393.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 398.05 | 391.28 | 393.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 399.60 | 392.95 | 394.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 401.75 | 392.95 | 394.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 393.75 | 394.06 | 394.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 393.75 | 394.06 | 394.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 395.75 | 394.40 | 394.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 396.00 | 394.40 | 394.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 390.70 | 393.66 | 394.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 398.55 | 393.66 | 394.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 392.40 | 392.60 | 393.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 393.65 | 392.60 | 393.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 391.50 | 392.28 | 393.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 387.90 | 391.30 | 392.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 386.75 | 390.71 | 392.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 387.70 | 390.71 | 392.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 368.50 | 379.84 | 385.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 368.31 | 379.84 | 385.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 367.41 | 377.30 | 383.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-23 13:15:00 | 349.11 | 358.36 | 369.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 286.35 | 283.58 | 283.27 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 278.45 | 282.70 | 282.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 274.70 | 281.10 | 282.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 256.15 | 255.87 | 262.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 256.15 | 255.87 | 262.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 263.85 | 258.02 | 262.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 263.35 | 258.02 | 262.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 263.10 | 259.04 | 262.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 264.30 | 259.04 | 262.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 263.00 | 260.52 | 262.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 270.25 | 260.52 | 262.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 272.00 | 262.82 | 263.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 272.20 | 262.82 | 263.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 272.85 | 264.82 | 264.11 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 263.15 | 266.14 | 266.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 260.65 | 264.43 | 265.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 269.05 | 257.99 | 260.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 269.05 | 257.99 | 260.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 261.05 | 258.60 | 260.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 259.65 | 260.37 | 260.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 260.00 | 260.37 | 260.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 264.10 | 261.23 | 261.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 264.10 | 261.23 | 261.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 09:15:00 | 278.30 | 264.64 | 262.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 15:15:00 | 298.80 | 299.25 | 290.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 09:15:00 | 295.00 | 299.25 | 290.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 294.45 | 298.29 | 290.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 297.00 | 295.17 | 292.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 297.00 | 298.62 | 295.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 298.15 | 298.17 | 296.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 297.10 | 297.37 | 296.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 296.20 | 297.13 | 296.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 296.35 | 297.13 | 296.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 296.40 | 296.99 | 296.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 13:30:00 | 296.30 | 296.99 | 296.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 295.80 | 296.75 | 296.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 295.45 | 296.75 | 296.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 295.15 | 296.43 | 296.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 295.15 | 296.43 | 296.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 288.65 | 294.87 | 295.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 300.20 | 292.98 | 293.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 300.50 | 292.98 | 293.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 300.85 | 294.56 | 294.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 304.15 | 296.48 | 295.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 310.25 | 310.64 | 307.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 305.35 | 310.64 | 307.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 305.65 | 309.64 | 307.24 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 302.00 | 306.03 | 306.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 296.30 | 303.01 | 304.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 293.00 | 291.27 | 294.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 294.10 | 291.27 | 294.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 296.05 | 292.22 | 294.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 295.50 | 292.22 | 294.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 297.20 | 293.22 | 294.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 296.70 | 293.22 | 294.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 297.50 | 294.07 | 295.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 296.00 | 295.50 | 295.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 297.20 | 295.84 | 295.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 297.20 | 295.84 | 295.72 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 295.15 | 295.66 | 295.66 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 295.85 | 295.70 | 295.68 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 294.90 | 295.54 | 295.61 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 296.75 | 295.78 | 295.71 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 294.45 | 295.80 | 295.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 290.45 | 294.73 | 295.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 293.40 | 293.05 | 293.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 293.20 | 293.27 | 294.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 292.40 | 293.10 | 293.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 291.55 | 292.70 | 293.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 293.60 | 292.35 | 292.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 293.60 | 292.35 | 292.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 296.80 | 293.24 | 292.73 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 474.95 | 2024-05-16 13:15:00 | 466.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-05-23 14:15:00 | 495.95 | 2024-05-27 10:15:00 | 483.95 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-05-23 14:45:00 | 498.00 | 2024-05-27 10:15:00 | 483.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-05-24 10:15:00 | 496.00 | 2024-05-27 10:15:00 | 483.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-05-24 11:00:00 | 497.25 | 2024-05-27 10:15:00 | 483.95 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-06-12 09:30:00 | 508.80 | 2024-06-14 12:15:00 | 505.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-06-12 11:30:00 | 508.95 | 2024-06-14 12:15:00 | 505.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-14 10:00:00 | 509.50 | 2024-06-14 12:15:00 | 505.10 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-06-14 11:30:00 | 508.70 | 2024-06-14 12:15:00 | 505.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-06-19 10:15:00 | 494.55 | 2024-06-20 09:15:00 | 506.40 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-06-19 12:15:00 | 495.15 | 2024-06-20 09:15:00 | 506.40 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-06-19 13:00:00 | 495.10 | 2024-06-20 09:15:00 | 506.40 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-06-26 12:15:00 | 505.15 | 2024-07-01 11:15:00 | 509.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-06-26 12:45:00 | 505.25 | 2024-07-01 11:15:00 | 509.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-06-26 14:00:00 | 505.00 | 2024-07-01 11:15:00 | 509.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-06-27 10:00:00 | 504.00 | 2024-07-01 11:15:00 | 509.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-23 12:15:00 | 513.55 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-07-23 13:00:00 | 522.75 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-07-23 14:15:00 | 523.00 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-07-23 14:45:00 | 524.55 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-07-24 11:30:00 | 531.25 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-07-24 12:45:00 | 531.35 | 2024-07-24 13:15:00 | 535.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-07-31 10:30:00 | 520.40 | 2024-08-05 09:15:00 | 494.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 11:30:00 | 520.35 | 2024-08-05 09:15:00 | 494.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 14:00:00 | 520.40 | 2024-08-05 09:15:00 | 494.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 09:45:00 | 517.70 | 2024-08-05 11:15:00 | 491.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 10:30:00 | 520.40 | 2024-08-06 11:15:00 | 500.60 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2024-07-31 11:30:00 | 520.35 | 2024-08-06 11:15:00 | 500.60 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2024-07-31 14:00:00 | 520.40 | 2024-08-06 11:15:00 | 500.60 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2024-08-01 09:45:00 | 517.70 | 2024-08-06 11:15:00 | 500.60 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2024-08-09 10:30:00 | 492.00 | 2024-08-12 15:15:00 | 494.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-08-09 12:00:00 | 492.00 | 2024-08-12 15:15:00 | 494.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-08-09 15:15:00 | 490.20 | 2024-08-16 10:15:00 | 494.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-08-12 09:45:00 | 490.90 | 2024-08-16 10:15:00 | 494.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-08-12 12:15:00 | 490.35 | 2024-08-16 10:15:00 | 494.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-08-12 13:00:00 | 490.80 | 2024-08-16 10:15:00 | 494.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-08-13 09:45:00 | 485.45 | 2024-08-16 10:15:00 | 494.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-08-22 09:15:00 | 508.50 | 2024-08-26 09:15:00 | 500.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-30 10:15:00 | 484.65 | 2024-09-09 09:15:00 | 460.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 10:45:00 | 484.90 | 2024-09-09 09:15:00 | 460.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 12:45:00 | 484.85 | 2024-09-09 09:15:00 | 460.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:30:00 | 483.95 | 2024-09-09 09:15:00 | 459.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 10:15:00 | 484.65 | 2024-09-09 15:15:00 | 465.00 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-08-30 10:45:00 | 484.90 | 2024-09-09 15:15:00 | 465.00 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2024-08-30 12:45:00 | 484.85 | 2024-09-09 15:15:00 | 465.00 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2024-09-02 09:30:00 | 483.95 | 2024-09-09 15:15:00 | 465.00 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-09-03 10:15:00 | 475.00 | 2024-09-10 12:15:00 | 481.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-05 10:45:00 | 474.45 | 2024-09-10 12:15:00 | 481.95 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-09-05 13:15:00 | 475.00 | 2024-09-10 12:15:00 | 481.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-05 14:00:00 | 475.00 | 2024-09-10 12:15:00 | 481.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-09 09:15:00 | 462.80 | 2024-09-10 12:15:00 | 481.95 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2024-09-12 11:15:00 | 494.80 | 2024-09-16 13:15:00 | 483.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-09-12 12:45:00 | 493.95 | 2024-09-16 13:15:00 | 483.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-09-13 09:45:00 | 492.90 | 2024-09-16 13:15:00 | 483.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-09-30 09:45:00 | 474.75 | 2024-10-08 09:15:00 | 451.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:45:00 | 474.75 | 2024-10-08 09:15:00 | 459.60 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2024-09-30 14:30:00 | 474.45 | 2024-10-08 09:15:00 | 450.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:30:00 | 474.45 | 2024-10-08 09:15:00 | 459.60 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-10-03 09:15:00 | 473.15 | 2024-10-08 09:15:00 | 449.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 473.15 | 2024-10-08 09:15:00 | 459.60 | STOP_HIT | 0.50 | 2.86% |
| BUY | retest2 | 2024-10-31 13:00:00 | 459.50 | 2024-11-04 11:15:00 | 452.65 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-10-31 13:30:00 | 459.15 | 2024-11-04 11:15:00 | 452.65 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-10-31 15:00:00 | 459.00 | 2024-11-04 11:15:00 | 452.65 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-06 09:15:00 | 461.40 | 2024-11-08 13:15:00 | 507.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-08 13:00:00 | 482.85 | 2024-11-13 09:15:00 | 466.40 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest1 | 2024-11-19 10:45:00 | 453.00 | 2024-11-22 15:15:00 | 447.50 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest1 | 2024-11-19 12:30:00 | 453.30 | 2024-11-22 15:15:00 | 447.50 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-11-25 11:30:00 | 445.30 | 2024-11-27 09:15:00 | 449.65 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-11-25 12:45:00 | 444.00 | 2024-11-27 09:15:00 | 449.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-11-26 09:30:00 | 445.80 | 2024-11-27 09:15:00 | 449.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-11-26 10:30:00 | 445.35 | 2024-11-27 09:15:00 | 449.65 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-11-29 09:45:00 | 453.05 | 2024-11-29 10:15:00 | 450.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-12-11 09:30:00 | 502.05 | 2024-12-17 09:15:00 | 501.70 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-12-13 10:45:00 | 497.00 | 2024-12-17 09:15:00 | 501.70 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-12-20 12:00:00 | 484.45 | 2024-12-27 10:15:00 | 484.25 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-12-27 10:00:00 | 480.50 | 2024-12-27 10:15:00 | 484.25 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-02-07 11:15:00 | 460.80 | 2025-02-07 14:15:00 | 447.85 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest1 | 2025-02-12 09:15:00 | 409.30 | 2025-02-19 09:15:00 | 407.75 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-02-14 10:45:00 | 413.75 | 2025-02-20 10:15:00 | 408.85 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-02-28 12:45:00 | 378.95 | 2025-03-03 11:15:00 | 360.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 12:45:00 | 378.95 | 2025-03-04 09:15:00 | 370.15 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-02-28 15:15:00 | 378.25 | 2025-03-04 09:15:00 | 359.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 15:15:00 | 378.25 | 2025-03-04 09:15:00 | 370.15 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2025-03-03 09:45:00 | 373.85 | 2025-03-05 10:15:00 | 385.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-03-07 10:15:00 | 390.35 | 2025-03-10 09:15:00 | 375.80 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-03-17 14:15:00 | 348.35 | 2025-03-18 09:15:00 | 362.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-03-17 15:00:00 | 348.40 | 2025-03-18 09:15:00 | 362.40 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest1 | 2025-03-21 09:15:00 | 375.95 | 2025-03-24 15:15:00 | 375.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-03-25 09:15:00 | 377.60 | 2025-03-25 09:15:00 | 372.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-03-28 12:00:00 | 361.25 | 2025-04-01 09:15:00 | 371.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-04-02 12:30:00 | 374.10 | 2025-04-04 09:15:00 | 363.00 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-04-02 13:00:00 | 373.65 | 2025-04-04 09:15:00 | 363.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-04-03 09:15:00 | 374.15 | 2025-04-04 09:15:00 | 363.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-04-03 10:30:00 | 375.90 | 2025-04-04 09:15:00 | 363.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-04-07 09:15:00 | 347.75 | 2025-04-09 15:15:00 | 363.50 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-04-23 09:15:00 | 402.65 | 2025-04-25 14:15:00 | 402.85 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-04-28 13:45:00 | 405.80 | 2025-04-28 15:15:00 | 409.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-04-28 14:15:00 | 406.00 | 2025-04-28 15:15:00 | 409.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-29 10:00:00 | 406.00 | 2025-05-02 13:15:00 | 408.55 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-04-30 10:15:00 | 405.70 | 2025-05-02 15:15:00 | 405.00 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-04-30 11:30:00 | 401.95 | 2025-05-02 15:15:00 | 405.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-06 09:15:00 | 393.00 | 2025-05-08 09:15:00 | 410.75 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-05-09 11:30:00 | 404.30 | 2025-05-19 14:15:00 | 420.00 | STOP_HIT | 1.00 | 3.88% |
| BUY | retest2 | 2025-05-29 09:15:00 | 422.45 | 2025-05-29 10:15:00 | 417.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-05-29 14:30:00 | 422.35 | 2025-05-30 09:15:00 | 415.20 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-06-03 13:15:00 | 413.35 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-06-03 14:00:00 | 414.30 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-06-04 09:15:00 | 414.45 | 2025-06-09 10:15:00 | 415.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-06-25 09:15:00 | 408.85 | 2025-07-03 14:15:00 | 449.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 11:00:00 | 408.50 | 2025-07-03 14:15:00 | 449.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-16 10:15:00 | 430.80 | 2025-07-17 10:15:00 | 435.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-30 09:30:00 | 414.40 | 2025-08-08 15:15:00 | 393.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:30:00 | 414.40 | 2025-08-12 09:15:00 | 394.95 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2025-08-18 10:15:00 | 403.70 | 2025-08-28 14:15:00 | 421.30 | STOP_HIT | 1.00 | 4.36% |
| BUY | retest2 | 2025-08-19 10:15:00 | 404.00 | 2025-08-28 14:15:00 | 421.30 | STOP_HIT | 1.00 | 4.28% |
| SELL | retest2 | 2025-09-02 13:15:00 | 415.30 | 2025-09-03 12:15:00 | 423.90 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-08 11:15:00 | 433.50 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-09-08 11:45:00 | 432.30 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-09-08 12:15:00 | 432.45 | 2025-09-09 11:15:00 | 424.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-10 11:30:00 | 425.25 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-09-10 14:45:00 | 425.20 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-09-10 15:15:00 | 424.10 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-09-11 10:15:00 | 425.15 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-09-11 11:30:00 | 422.90 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-09-17 09:30:00 | 422.80 | 2025-09-18 09:15:00 | 423.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-26 09:15:00 | 402.55 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-09-26 10:00:00 | 406.05 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-26 13:15:00 | 407.05 | 2025-09-30 09:15:00 | 412.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2025-10-03 11:15:00 | 414.50 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2025-10-03 12:45:00 | 414.90 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest1 | 2025-10-03 14:15:00 | 414.60 | 2025-10-08 10:15:00 | 418.95 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-10-09 09:15:00 | 426.70 | 2025-10-14 13:15:00 | 428.80 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-10-24 09:15:00 | 436.35 | 2025-10-24 10:15:00 | 432.25 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-24 10:00:00 | 435.25 | 2025-10-24 10:15:00 | 432.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 10:30:00 | 420.00 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-28 11:15:00 | 419.95 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-29 13:15:00 | 419.25 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-29 14:00:00 | 420.25 | 2025-10-30 14:15:00 | 422.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-27 11:15:00 | 496.55 | 2025-11-27 11:15:00 | 490.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-19 11:30:00 | 482.15 | 2025-12-22 13:15:00 | 486.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-29 10:30:00 | 458.50 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-29 11:00:00 | 458.25 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 12:15:00 | 458.10 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-30 09:15:00 | 457.25 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-01 12:00:00 | 457.45 | 2026-01-02 12:15:00 | 461.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-13 12:00:00 | 438.05 | 2026-01-20 10:15:00 | 416.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 437.75 | 2026-01-20 10:15:00 | 415.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 437.65 | 2026-01-20 10:15:00 | 415.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 437.20 | 2026-01-20 11:15:00 | 415.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 438.05 | 2026-01-21 10:15:00 | 394.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 437.75 | 2026-01-21 10:15:00 | 393.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 437.65 | 2026-01-21 10:15:00 | 393.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 437.20 | 2026-01-21 10:15:00 | 393.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 399.75 | 2026-01-23 13:15:00 | 379.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 399.75 | 2026-01-27 12:15:00 | 395.85 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2026-01-28 12:15:00 | 397.15 | 2026-01-28 14:15:00 | 402.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-02-19 11:30:00 | 387.90 | 2026-02-20 11:15:00 | 368.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 386.75 | 2026-02-20 11:15:00 | 368.31 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-02-19 13:15:00 | 387.70 | 2026-02-20 12:15:00 | 367.41 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-02-19 11:30:00 | 387.90 | 2026-02-23 13:15:00 | 349.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 386.75 | 2026-02-23 13:15:00 | 348.93 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-02-19 13:15:00 | 387.70 | 2026-02-24 09:15:00 | 348.07 | TARGET_HIT | 0.50 | 10.22% |
| SELL | retest2 | 2026-04-01 13:30:00 | 259.65 | 2026-04-01 15:15:00 | 264.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-01 14:15:00 | 260.00 | 2026-04-01 15:15:00 | 264.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-04-08 09:30:00 | 297.00 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-09 10:15:00 | 297.00 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-09 14:45:00 | 298.15 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-10 11:30:00 | 297.10 | 2026-04-10 15:15:00 | 295.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-28 09:45:00 | 296.00 | 2026-04-28 10:15:00 | 297.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-05-04 10:45:00 | 293.20 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-05-04 12:00:00 | 292.40 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-05-05 09:15:00 | 291.55 | 2026-05-06 11:15:00 | 293.60 | STOP_HIT | 1.00 | -0.70% |
