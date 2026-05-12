# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 743.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 136 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 47 |
| ALERT3 | 284 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 100 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 95 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 62
- **Target hits / Stop hits / Partials:** 10 / 95 / 19
- **Avg / median % per leg:** 1.12% / 0.01%
- **Sum % (uncompounded):** 139.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 20 | 39.2% | 3 | 48 | 0 | 0.08% | 3.8% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.23% | -0.7% |
| BUY @ 3rd Alert (retest2) | 48 | 19 | 39.6% | 3 | 45 | 0 | 0.09% | 4.5% |
| SELL (all) | 73 | 42 | 57.5% | 7 | 47 | 19 | 1.86% | 135.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.99% | -4.0% |
| SELL @ 3rd Alert (retest2) | 71 | 42 | 59.2% | 7 | 45 | 19 | 1.97% | 139.6% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.93% | -4.7% |
| retest2 (combined) | 119 | 61 | 51.3% | 10 | 90 | 19 | 1.21% | 144.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 15:15:00 | 518.00 | 520.84 | 520.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 09:15:00 | 515.95 | 519.86 | 520.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 529.90 | 517.26 | 518.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 529.90 | 517.26 | 518.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 529.90 | 517.26 | 518.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 529.90 | 517.26 | 518.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 536.85 | 521.18 | 519.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 548.80 | 526.70 | 522.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 537.10 | 540.11 | 532.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 537.10 | 540.11 | 532.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 536.60 | 539.41 | 533.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 535.30 | 539.41 | 533.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 532.25 | 537.93 | 533.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 533.05 | 537.93 | 533.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 539.20 | 538.18 | 534.03 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 13:15:00 | 526.50 | 532.08 | 532.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 15:15:00 | 525.80 | 529.98 | 531.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 534.05 | 530.80 | 531.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 534.05 | 530.80 | 531.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 534.05 | 530.80 | 531.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 535.60 | 530.80 | 531.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 530.00 | 530.64 | 531.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 533.95 | 530.64 | 531.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 521.95 | 522.50 | 525.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:15:00 | 521.00 | 522.50 | 525.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 520.40 | 522.08 | 525.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 11:30:00 | 518.80 | 521.66 | 524.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 13:45:00 | 518.10 | 520.66 | 523.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 517.95 | 519.41 | 522.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:30:00 | 516.60 | 519.10 | 521.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 514.90 | 517.72 | 520.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 515.00 | 517.72 | 520.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 517.30 | 517.57 | 519.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 513.40 | 516.19 | 518.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 492.86 | 500.15 | 502.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 492.19 | 500.15 | 502.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 492.05 | 500.15 | 502.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 506.00 | 501.32 | 502.64 | SL hit (close>ema200) qty=0.50 sl=501.32 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 506.40 | 502.25 | 502.12 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 15:15:00 | 499.70 | 502.02 | 502.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 491.85 | 499.99 | 501.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 489.80 | 484.11 | 489.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 489.80 | 484.11 | 489.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 489.80 | 484.11 | 489.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 489.80 | 484.11 | 489.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 490.25 | 485.34 | 489.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 490.00 | 485.34 | 489.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 492.15 | 486.70 | 489.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:30:00 | 489.00 | 486.70 | 489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 490.15 | 487.39 | 490.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 490.15 | 487.39 | 490.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 483.55 | 487.01 | 489.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 496.90 | 487.01 | 489.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 496.70 | 488.95 | 490.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 497.30 | 488.95 | 490.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 500.05 | 491.17 | 490.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 505.15 | 498.95 | 495.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 500.55 | 501.37 | 498.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 500.55 | 501.37 | 498.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 500.55 | 501.21 | 498.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 509.45 | 501.21 | 498.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 512.75 | 515.92 | 516.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 512.75 | 515.92 | 516.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 511.05 | 514.95 | 515.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 514.70 | 511.30 | 513.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 514.70 | 511.30 | 513.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 514.70 | 511.30 | 513.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 514.70 | 511.30 | 513.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 515.25 | 512.09 | 513.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:30:00 | 516.50 | 512.09 | 513.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 512.50 | 512.17 | 513.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:30:00 | 513.25 | 512.17 | 513.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 516.15 | 512.97 | 513.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 516.15 | 512.97 | 513.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 515.50 | 513.48 | 513.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 524.40 | 513.48 | 513.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 515.95 | 513.97 | 513.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 12:15:00 | 536.25 | 519.84 | 516.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 545.40 | 546.53 | 536.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:30:00 | 544.90 | 546.53 | 536.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 535.70 | 541.91 | 538.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 535.70 | 541.91 | 538.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 539.90 | 541.50 | 538.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:15:00 | 535.50 | 541.50 | 538.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 535.80 | 540.36 | 537.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 536.60 | 540.36 | 537.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 537.00 | 539.69 | 537.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 530.10 | 539.69 | 537.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 527.10 | 535.29 | 536.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 11:15:00 | 526.95 | 533.63 | 535.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 10:15:00 | 531.95 | 529.95 | 532.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 11:00:00 | 531.95 | 529.95 | 532.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 534.05 | 530.77 | 532.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:00:00 | 529.95 | 531.24 | 532.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 529.85 | 530.91 | 532.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:45:00 | 529.25 | 530.34 | 531.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:15:00 | 530.00 | 530.53 | 531.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 525.50 | 529.53 | 531.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-26 15:15:00 | 535.00 | 531.67 | 531.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 535.00 | 531.67 | 531.58 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 526.50 | 530.69 | 531.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 520.05 | 526.86 | 529.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 526.40 | 524.08 | 526.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 526.40 | 524.08 | 526.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 526.40 | 524.08 | 526.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 526.50 | 524.08 | 526.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 521.55 | 523.58 | 526.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 527.30 | 523.58 | 526.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 517.00 | 520.38 | 523.76 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 535.70 | 526.61 | 525.53 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 522.50 | 525.60 | 526.00 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 527.80 | 526.25 | 526.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 528.30 | 526.54 | 526.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 10:15:00 | 535.40 | 536.29 | 532.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 13:15:00 | 533.20 | 535.85 | 533.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 533.20 | 535.85 | 533.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 533.20 | 535.85 | 533.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 536.25 | 535.93 | 533.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 537.80 | 535.93 | 533.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 529.80 | 534.23 | 533.36 | SL hit (close<static) qty=1.00 sl=530.80 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 582.75 | 588.21 | 588.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 559.95 | 581.92 | 585.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 09:15:00 | 578.65 | 570.10 | 575.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 578.65 | 570.10 | 575.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 578.65 | 570.10 | 575.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 576.95 | 570.10 | 575.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 584.25 | 572.93 | 576.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 604.20 | 572.93 | 576.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 583.05 | 574.96 | 577.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:30:00 | 587.70 | 574.96 | 577.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 566.45 | 556.05 | 560.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 569.85 | 556.05 | 560.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 575.95 | 560.03 | 562.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 575.95 | 560.03 | 562.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 580.90 | 564.20 | 563.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 584.00 | 576.82 | 571.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 597.40 | 597.79 | 590.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 608.05 | 597.79 | 590.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 610.20 | 609.30 | 602.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 602.00 | 609.30 | 602.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 609.90 | 608.93 | 605.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 604.10 | 608.93 | 605.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 604.00 | 607.95 | 605.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 604.00 | 607.95 | 605.23 | SL hit (close<ema400) qty=1.00 sl=605.23 alert=retest1 |

### Cycle 17 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 614.15 | 619.64 | 619.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 605.00 | 614.24 | 616.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 613.40 | 612.60 | 615.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 613.40 | 612.60 | 615.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 613.40 | 612.60 | 615.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 615.65 | 612.60 | 615.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 623.50 | 614.78 | 616.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 623.50 | 614.78 | 616.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 646.50 | 621.13 | 618.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 652.50 | 627.40 | 621.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 656.15 | 657.37 | 645.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 656.15 | 657.37 | 645.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 681.00 | 695.30 | 685.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 681.00 | 695.30 | 685.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 685.80 | 693.40 | 685.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 674.40 | 693.40 | 685.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 672.85 | 689.29 | 684.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 672.85 | 689.29 | 684.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 671.15 | 685.66 | 683.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 670.80 | 685.66 | 683.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 662.65 | 678.87 | 680.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 644.65 | 669.81 | 675.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 659.65 | 657.06 | 665.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:15:00 | 647.35 | 657.06 | 665.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:30:00 | 649.00 | 652.37 | 660.84 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 661.05 | 650.15 | 656.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 661.05 | 650.15 | 656.11 | SL hit (close>ema400) qty=1.00 sl=656.11 alert=retest1 |

### Cycle 20 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 662.45 | 645.49 | 643.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 666.25 | 649.64 | 645.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 676.00 | 676.76 | 667.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 14:45:00 | 676.30 | 676.76 | 667.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 677.40 | 677.25 | 669.41 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 664.35 | 668.06 | 668.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 658.15 | 666.07 | 667.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 657.30 | 648.14 | 652.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 657.30 | 648.14 | 652.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 657.30 | 648.14 | 652.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 655.45 | 648.14 | 652.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 654.80 | 649.47 | 653.04 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 667.25 | 656.98 | 655.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 718.65 | 679.32 | 670.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 723.00 | 723.19 | 703.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:30:00 | 723.25 | 723.19 | 703.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 720.05 | 727.72 | 717.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:30:00 | 722.00 | 727.72 | 717.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 717.75 | 725.72 | 717.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 717.75 | 725.72 | 717.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 715.20 | 723.62 | 717.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:45:00 | 714.30 | 723.62 | 717.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 715.90 | 722.08 | 717.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 713.30 | 722.08 | 717.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 713.50 | 718.11 | 716.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 713.50 | 718.11 | 716.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 711.90 | 716.87 | 716.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 710.20 | 716.87 | 716.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 14:15:00 | 704.55 | 714.41 | 715.29 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 739.25 | 717.09 | 716.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 751.90 | 724.05 | 719.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 740.95 | 742.21 | 732.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 11:00:00 | 740.95 | 742.21 | 732.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 725.60 | 738.81 | 735.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 725.60 | 738.81 | 735.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 727.30 | 736.51 | 734.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 718.00 | 736.51 | 734.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 725.65 | 733.13 | 733.33 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 739.35 | 732.57 | 732.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 754.50 | 736.96 | 734.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 737.10 | 738.91 | 735.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 14:00:00 | 737.10 | 738.91 | 735.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 730.10 | 737.15 | 735.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 730.10 | 737.15 | 735.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 737.00 | 737.12 | 735.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 727.95 | 737.12 | 735.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 732.90 | 736.28 | 735.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:30:00 | 732.80 | 736.28 | 735.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 735.50 | 736.12 | 735.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:00:00 | 772.90 | 743.48 | 738.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 15:15:00 | 759.95 | 761.98 | 762.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 759.95 | 761.98 | 762.02 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 767.05 | 763.00 | 762.48 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 15:15:00 | 754.70 | 762.52 | 763.32 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 772.30 | 762.64 | 762.25 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 09:15:00 | 755.85 | 765.11 | 765.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 744.10 | 760.90 | 763.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 12:15:00 | 742.30 | 741.67 | 749.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 12:15:00 | 742.30 | 741.67 | 749.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 742.30 | 741.67 | 749.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 742.30 | 741.67 | 749.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 772.00 | 747.73 | 751.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 772.00 | 747.73 | 751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 774.00 | 752.99 | 753.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:15:00 | 771.00 | 752.99 | 753.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 771.00 | 756.59 | 755.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 09:15:00 | 824.50 | 782.40 | 770.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 801.85 | 805.50 | 791.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 801.85 | 805.50 | 791.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 801.85 | 805.50 | 791.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 797.90 | 805.50 | 791.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 789.15 | 800.66 | 792.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 789.15 | 800.66 | 792.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 793.30 | 799.19 | 792.72 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 740.05 | 784.72 | 787.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 728.90 | 773.56 | 782.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 742.25 | 737.80 | 748.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 10:00:00 | 742.25 | 737.80 | 748.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 738.55 | 735.40 | 741.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 740.00 | 735.40 | 741.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 735.50 | 735.08 | 740.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 13:15:00 | 734.00 | 735.08 | 740.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 14:30:00 | 733.60 | 735.01 | 739.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 732.50 | 735.10 | 738.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 13:00:00 | 733.00 | 733.30 | 736.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 733.60 | 733.36 | 736.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:45:00 | 735.25 | 733.36 | 736.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 735.55 | 733.80 | 736.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:45:00 | 736.35 | 733.80 | 736.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 736.00 | 734.24 | 736.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 734.00 | 734.24 | 736.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 724.50 | 732.29 | 735.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:15:00 | 720.00 | 732.29 | 735.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 12:15:00 | 740.00 | 734.09 | 733.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 12:15:00 | 740.00 | 734.09 | 733.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 747.30 | 736.73 | 735.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 766.80 | 771.42 | 758.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 766.80 | 771.42 | 758.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 757.20 | 767.24 | 759.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 757.20 | 767.24 | 759.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 763.00 | 766.39 | 759.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 760.85 | 766.39 | 759.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 750.50 | 763.21 | 758.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 750.50 | 763.21 | 758.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 755.00 | 761.57 | 758.32 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 741.00 | 755.97 | 756.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 726.90 | 744.74 | 749.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 710.05 | 708.02 | 719.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 710.05 | 708.02 | 719.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 685.70 | 702.71 | 713.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:45:00 | 666.85 | 684.00 | 697.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:30:00 | 667.95 | 672.54 | 686.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 664.00 | 673.47 | 677.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:15:00 | 666.15 | 673.85 | 677.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 681.90 | 675.46 | 677.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 681.90 | 675.46 | 677.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 684.20 | 677.21 | 678.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 684.20 | 677.21 | 678.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 701.75 | 683.21 | 680.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 701.75 | 683.21 | 680.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 711.55 | 696.01 | 687.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 714.65 | 714.73 | 706.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 714.65 | 714.73 | 706.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 710.60 | 713.90 | 706.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:30:00 | 711.10 | 713.90 | 706.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 708.55 | 712.83 | 707.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 708.55 | 712.83 | 707.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 717.75 | 713.81 | 708.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 720.05 | 713.81 | 708.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 722.20 | 715.95 | 711.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 720.75 | 717.71 | 713.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:00:00 | 723.95 | 718.96 | 714.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 742.95 | 745.10 | 738.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 12:15:00 | 721.85 | 734.67 | 735.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 721.85 | 734.67 | 735.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 714.95 | 730.72 | 733.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 674.90 | 673.38 | 685.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 674.90 | 673.38 | 685.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 676.05 | 662.75 | 667.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 676.05 | 662.75 | 667.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 673.10 | 664.82 | 668.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:15:00 | 680.25 | 664.82 | 668.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 663.00 | 667.12 | 668.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:45:00 | 665.80 | 667.12 | 668.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 666.00 | 666.56 | 668.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 667.25 | 666.56 | 668.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 658.95 | 665.03 | 667.36 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 679.25 | 669.91 | 668.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 680.95 | 672.12 | 669.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 743.05 | 745.15 | 731.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:00:00 | 743.05 | 745.15 | 731.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 743.70 | 743.00 | 734.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 752.90 | 744.31 | 736.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 782.35 | 786.95 | 787.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 782.35 | 786.95 | 787.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 768.75 | 782.44 | 785.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 13:15:00 | 777.20 | 775.21 | 779.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 13:15:00 | 777.20 | 775.21 | 779.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 777.20 | 775.21 | 779.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 778.75 | 775.21 | 779.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 791.40 | 776.90 | 778.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 791.40 | 776.90 | 778.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 792.55 | 780.03 | 779.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 808.00 | 788.56 | 784.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 843.85 | 848.30 | 829.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 837.05 | 848.30 | 829.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 829.50 | 844.54 | 829.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 829.80 | 844.54 | 829.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 827.80 | 841.19 | 829.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 829.10 | 841.19 | 829.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 823.10 | 837.57 | 828.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:45:00 | 826.35 | 837.57 | 828.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 817.55 | 833.57 | 827.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 818.80 | 833.57 | 827.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 862.40 | 846.76 | 839.12 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 823.65 | 837.34 | 838.19 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 12:15:00 | 842.30 | 839.27 | 838.87 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 834.45 | 838.46 | 838.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 810.25 | 832.23 | 835.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 11:15:00 | 797.85 | 780.86 | 794.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 11:15:00 | 797.85 | 780.86 | 794.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 797.85 | 780.86 | 794.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 797.00 | 780.86 | 794.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 797.00 | 784.09 | 794.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:45:00 | 797.65 | 784.09 | 794.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 794.15 | 786.10 | 794.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:15:00 | 802.95 | 786.10 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 801.85 | 789.25 | 795.20 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 836.75 | 800.47 | 799.37 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 800.00 | 803.28 | 803.56 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 811.10 | 804.70 | 803.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 09:15:00 | 826.65 | 811.98 | 807.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 15:15:00 | 822.00 | 822.38 | 815.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:15:00 | 830.45 | 822.38 | 815.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 816.10 | 821.84 | 816.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 816.10 | 821.84 | 816.82 | SL hit (close<ema400) qty=1.00 sl=816.82 alert=retest1 |

### Cycle 47 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 814.95 | 822.92 | 823.00 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 14:15:00 | 861.65 | 828.29 | 825.19 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 822.05 | 836.84 | 836.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 817.30 | 832.93 | 835.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 719.05 | 714.52 | 728.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:30:00 | 720.80 | 714.52 | 728.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 705.25 | 713.56 | 725.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:45:00 | 701.45 | 709.78 | 714.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 11:45:00 | 698.80 | 706.31 | 711.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 695.45 | 705.32 | 709.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 666.38 | 684.73 | 695.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 663.86 | 684.73 | 695.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 660.68 | 684.73 | 695.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 689.55 | 674.63 | 683.10 | SL hit (close>ema200) qty=0.50 sl=674.63 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 706.30 | 687.98 | 687.03 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 670.75 | 687.07 | 688.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 663.10 | 678.26 | 683.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 669.45 | 664.86 | 673.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 669.45 | 664.86 | 673.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 676.55 | 667.20 | 674.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 676.55 | 667.20 | 674.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 675.00 | 668.76 | 674.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 673.00 | 668.76 | 674.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 658.10 | 666.63 | 672.69 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 684.90 | 674.31 | 673.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 711.35 | 683.67 | 678.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 11:15:00 | 696.60 | 697.46 | 690.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 12:00:00 | 696.60 | 697.46 | 690.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 698.30 | 697.63 | 691.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:00:00 | 699.45 | 697.57 | 692.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 701.30 | 697.86 | 693.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 699.55 | 699.06 | 694.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:45:00 | 699.05 | 697.39 | 694.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 700.00 | 697.91 | 695.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 698.40 | 697.91 | 695.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 717.60 | 719.61 | 712.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 717.60 | 719.61 | 712.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 716.75 | 720.16 | 714.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:30:00 | 713.35 | 720.16 | 714.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 726.15 | 735.53 | 730.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 727.25 | 735.53 | 730.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 723.80 | 733.18 | 730.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 718.00 | 733.18 | 730.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 740.00 | 732.35 | 730.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 714.35 | 727.81 | 729.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 714.35 | 727.81 | 729.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 712.50 | 724.75 | 728.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 11:15:00 | 632.05 | 630.98 | 646.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 11:30:00 | 631.90 | 630.98 | 646.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 644.00 | 634.40 | 644.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 644.00 | 634.40 | 644.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 650.00 | 637.52 | 644.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 640.50 | 637.52 | 644.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 639.35 | 637.88 | 644.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:30:00 | 633.10 | 637.19 | 642.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 667.40 | 646.16 | 645.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 667.40 | 646.16 | 645.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 671.50 | 651.23 | 647.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 673.70 | 677.73 | 670.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 673.70 | 677.73 | 670.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 674.95 | 677.17 | 670.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 675.00 | 677.17 | 670.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 670.75 | 675.31 | 671.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 668.25 | 675.31 | 671.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 681.45 | 676.54 | 672.16 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 663.10 | 672.11 | 672.21 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 681.45 | 673.98 | 673.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 12:15:00 | 689.60 | 679.30 | 675.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 09:15:00 | 668.30 | 679.81 | 677.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 668.30 | 679.81 | 677.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 668.30 | 679.81 | 677.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 668.30 | 679.81 | 677.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 668.80 | 677.61 | 676.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:30:00 | 664.00 | 677.61 | 676.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 657.85 | 673.66 | 675.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 654.20 | 669.76 | 673.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 604.25 | 589.60 | 608.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 604.25 | 589.60 | 608.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 604.25 | 589.60 | 608.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 604.25 | 589.60 | 608.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 607.35 | 593.15 | 608.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 607.35 | 593.15 | 608.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 600.75 | 594.67 | 608.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 605.45 | 594.67 | 608.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 606.85 | 598.06 | 607.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 606.85 | 598.06 | 607.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 605.20 | 599.49 | 607.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:30:00 | 606.55 | 599.49 | 607.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 599.90 | 599.57 | 606.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 619.40 | 599.57 | 606.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 617.40 | 603.14 | 607.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 621.90 | 603.14 | 607.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 622.85 | 607.08 | 608.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 622.85 | 607.08 | 608.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 629.25 | 613.42 | 611.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 639.90 | 624.55 | 617.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 679.00 | 679.87 | 665.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 679.00 | 679.87 | 665.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 663.50 | 673.58 | 666.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 651.35 | 673.58 | 666.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 648.00 | 668.47 | 665.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:45:00 | 649.50 | 668.47 | 665.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 663.10 | 667.01 | 664.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:30:00 | 657.95 | 667.01 | 664.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 666.05 | 666.81 | 665.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:30:00 | 667.40 | 666.81 | 665.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 672.80 | 668.01 | 665.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:00:00 | 675.45 | 670.27 | 668.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:45:00 | 675.95 | 671.22 | 668.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:00:00 | 677.85 | 672.69 | 669.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:45:00 | 681.45 | 674.80 | 671.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 677.30 | 675.52 | 672.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-17 15:15:00 | 669.00 | 671.70 | 671.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 15:15:00 | 669.00 | 671.70 | 671.97 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 682.65 | 673.89 | 672.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 688.90 | 678.46 | 675.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 689.90 | 691.23 | 686.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 689.90 | 691.23 | 686.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 689.90 | 691.23 | 686.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 689.90 | 691.23 | 686.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 713.10 | 695.55 | 689.74 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 682.55 | 692.95 | 694.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 677.40 | 685.49 | 689.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 682.00 | 680.59 | 685.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 682.00 | 680.59 | 685.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 682.00 | 680.59 | 685.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 14:00:00 | 660.35 | 675.36 | 681.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 657.10 | 667.41 | 674.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 672.30 | 659.59 | 658.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 672.30 | 659.59 | 658.87 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 651.80 | 660.82 | 661.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 640.95 | 654.43 | 657.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 616.20 | 610.08 | 624.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 616.20 | 610.08 | 624.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 616.20 | 610.08 | 624.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 601.15 | 614.45 | 621.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 632.20 | 615.53 | 614.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 632.20 | 615.53 | 614.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 643.00 | 625.19 | 619.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 665.45 | 670.01 | 655.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 665.45 | 670.01 | 655.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 656.80 | 665.59 | 657.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:30:00 | 656.60 | 665.59 | 657.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 652.20 | 662.92 | 656.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 652.20 | 662.92 | 656.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 659.00 | 662.13 | 656.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 659.85 | 661.21 | 657.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:15:00 | 660.40 | 661.21 | 657.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 661.80 | 660.58 | 657.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 670.20 | 678.87 | 679.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 670.20 | 678.87 | 679.15 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 12:15:00 | 686.30 | 679.44 | 679.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 13:15:00 | 710.10 | 685.57 | 682.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 09:15:00 | 685.80 | 687.54 | 684.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 685.80 | 687.54 | 684.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 685.80 | 687.54 | 684.03 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 14:15:00 | 676.30 | 682.66 | 682.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 15:15:00 | 672.05 | 680.54 | 681.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 676.45 | 654.49 | 661.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 676.45 | 654.49 | 661.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 676.45 | 654.49 | 661.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 677.30 | 654.49 | 661.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 679.00 | 659.39 | 663.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 682.30 | 659.39 | 663.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 683.00 | 667.72 | 666.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 13:15:00 | 692.00 | 672.57 | 668.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 693.40 | 699.25 | 691.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 12:15:00 | 693.40 | 699.25 | 691.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 693.40 | 699.25 | 691.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 693.40 | 699.25 | 691.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 689.65 | 697.33 | 691.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 689.65 | 697.33 | 691.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 687.70 | 695.41 | 690.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 687.70 | 695.41 | 690.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 693.10 | 691.60 | 689.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 699.20 | 693.99 | 691.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 671.10 | 687.67 | 689.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 671.10 | 687.67 | 689.75 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 692.50 | 689.59 | 689.55 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 688.00 | 689.27 | 689.41 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 710.30 | 693.48 | 691.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 723.00 | 699.38 | 694.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 706.15 | 719.83 | 711.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 706.15 | 719.83 | 711.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 706.15 | 719.83 | 711.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 706.15 | 719.83 | 711.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 711.10 | 718.08 | 711.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 728.00 | 718.08 | 711.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 14:15:00 | 702.00 | 714.86 | 710.23 | SL hit (close<static) qty=1.00 sl=706.20 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 684.00 | 702.95 | 705.52 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 697.50 | 693.52 | 693.26 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 685.25 | 692.67 | 693.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 676.50 | 689.44 | 692.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 682.00 | 678.82 | 683.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 682.00 | 678.82 | 683.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 682.00 | 678.82 | 683.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:15:00 | 683.15 | 678.82 | 683.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 682.95 | 679.65 | 683.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 684.30 | 679.65 | 683.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 677.10 | 679.14 | 682.46 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 683.80 | 681.82 | 681.80 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 675.65 | 681.99 | 682.35 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 720.55 | 687.47 | 684.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 728.25 | 711.22 | 705.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 715.65 | 723.81 | 717.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 715.65 | 723.81 | 717.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 715.65 | 723.81 | 717.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 715.65 | 723.81 | 717.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 715.80 | 722.21 | 717.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 716.35 | 722.21 | 717.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 715.30 | 720.83 | 717.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 715.30 | 720.83 | 717.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 714.00 | 719.46 | 716.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 709.90 | 718.10 | 716.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 717.40 | 717.96 | 716.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 726.40 | 719.65 | 717.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 713.20 | 720.85 | 721.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 713.20 | 720.85 | 721.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 704.30 | 717.54 | 719.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 707.05 | 707.02 | 710.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 15:00:00 | 707.05 | 707.02 | 710.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 705.00 | 706.23 | 709.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 701.80 | 705.34 | 708.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 14:15:00 | 666.71 | 684.04 | 693.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 691.05 | 685.28 | 692.46 | SL hit (close>ema200) qty=0.50 sl=685.28 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 09:15:00 | 742.20 | 695.17 | 693.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 10:15:00 | 751.00 | 706.33 | 698.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 747.70 | 764.54 | 749.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 747.70 | 764.54 | 749.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 747.70 | 764.54 | 749.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 747.70 | 764.54 | 749.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 742.60 | 760.16 | 748.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 742.50 | 760.16 | 748.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 725.00 | 742.56 | 743.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 723.00 | 738.65 | 741.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 736.95 | 726.73 | 732.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 736.95 | 726.73 | 732.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 736.95 | 726.73 | 732.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 736.90 | 726.73 | 732.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 717.80 | 724.94 | 730.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 709.40 | 720.57 | 727.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 711.00 | 716.84 | 724.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 711.80 | 711.10 | 719.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 723.90 | 716.70 | 716.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 723.90 | 716.70 | 716.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 729.60 | 720.38 | 718.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 10:15:00 | 758.75 | 761.31 | 750.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:45:00 | 758.80 | 761.31 | 750.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 749.35 | 757.28 | 752.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 749.85 | 757.28 | 752.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 749.95 | 755.81 | 751.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 769.25 | 755.81 | 751.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 746.15 | 757.73 | 755.75 | SL hit (close<static) qty=1.00 sl=747.90 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 749.00 | 754.52 | 754.55 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 765.65 | 755.19 | 754.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 12:15:00 | 772.80 | 762.14 | 758.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 781.50 | 783.68 | 778.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 11:30:00 | 792.60 | 786.98 | 780.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 805.80 | 808.06 | 801.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 806.20 | 808.06 | 801.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 804.90 | 807.43 | 802.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 802.75 | 807.43 | 802.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 804.45 | 806.83 | 802.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 813.15 | 806.83 | 802.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 806.10 | 812.07 | 808.75 | SL hit (close<ema400) qty=1.00 sl=808.75 alert=retest1 |

### Cycle 85 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 825.00 | 830.24 | 830.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 807.90 | 819.28 | 824.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 816.60 | 811.04 | 816.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 816.60 | 811.04 | 816.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 816.60 | 811.04 | 816.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 816.15 | 811.04 | 816.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 824.60 | 813.75 | 817.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 824.60 | 813.75 | 817.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 825.70 | 816.14 | 818.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 823.35 | 816.14 | 818.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 822.95 | 819.12 | 819.05 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 816.40 | 818.57 | 818.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 805.30 | 815.92 | 817.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 796.70 | 794.14 | 802.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 796.70 | 794.14 | 802.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 796.10 | 791.50 | 797.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 796.10 | 791.50 | 797.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 787.00 | 790.60 | 796.52 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 807.65 | 799.24 | 798.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 15:15:00 | 815.00 | 805.96 | 802.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 805.65 | 806.37 | 803.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:15:00 | 807.50 | 806.37 | 803.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 805.95 | 806.29 | 803.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 817.65 | 806.29 | 803.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 785.00 | 801.57 | 802.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 785.00 | 801.57 | 802.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 778.95 | 797.04 | 799.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 709.00 | 708.31 | 719.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 707.80 | 708.31 | 719.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 698.60 | 694.74 | 699.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 698.60 | 694.74 | 699.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 699.90 | 695.77 | 699.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 702.50 | 695.77 | 699.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 695.35 | 695.69 | 699.46 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 702.85 | 700.45 | 700.41 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 698.80 | 700.12 | 700.27 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 702.40 | 700.40 | 700.35 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 699.35 | 700.19 | 700.26 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 701.90 | 700.53 | 700.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 702.65 | 700.95 | 700.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 701.10 | 701.32 | 700.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 701.10 | 701.32 | 700.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 701.10 | 701.32 | 700.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 698.80 | 701.32 | 700.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 700.90 | 701.24 | 700.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 696.40 | 701.24 | 700.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 703.25 | 701.64 | 701.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:15:00 | 700.35 | 701.64 | 701.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 697.45 | 700.80 | 700.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 697.45 | 700.80 | 700.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 706.90 | 702.02 | 701.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 707.60 | 703.32 | 702.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 711.00 | 703.32 | 702.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 703.80 | 703.96 | 703.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 703.80 | 703.96 | 703.97 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 705.20 | 704.00 | 703.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 711.35 | 705.47 | 704.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 727.35 | 727.62 | 722.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:00:00 | 727.35 | 727.62 | 722.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 726.85 | 727.85 | 723.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:45:00 | 731.00 | 728.28 | 724.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 14:30:00 | 728.65 | 728.49 | 725.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 15:00:00 | 729.30 | 728.49 | 725.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 731.00 | 728.35 | 726.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 729.40 | 729.42 | 727.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:00:00 | 729.40 | 729.42 | 727.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 725.75 | 728.68 | 727.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:30:00 | 725.05 | 728.68 | 727.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 724.00 | 727.75 | 727.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 728.25 | 727.75 | 727.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 734.95 | 731.87 | 729.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 724.60 | 730.44 | 730.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 724.60 | 730.44 | 730.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 715.75 | 722.11 | 726.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 720.90 | 720.73 | 724.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:45:00 | 722.35 | 720.73 | 724.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 738.00 | 724.51 | 725.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 738.00 | 724.51 | 725.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 721.00 | 723.81 | 724.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 718.20 | 722.01 | 723.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:45:00 | 718.00 | 714.60 | 717.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 716.70 | 715.87 | 717.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 717.90 | 711.06 | 710.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 717.90 | 711.06 | 710.36 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 709.55 | 712.77 | 712.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 704.50 | 709.89 | 711.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 691.00 | 689.60 | 697.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 691.00 | 689.60 | 697.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 670.00 | 665.82 | 670.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 670.00 | 665.82 | 670.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 670.45 | 666.75 | 670.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 670.00 | 666.75 | 670.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 670.00 | 667.40 | 670.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 655.15 | 667.40 | 670.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 662.00 | 646.89 | 644.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 662.00 | 646.89 | 644.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 664.25 | 654.86 | 649.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 653.90 | 660.21 | 654.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 653.90 | 660.21 | 654.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 653.90 | 660.21 | 654.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 653.75 | 660.21 | 654.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 652.55 | 658.68 | 654.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:15:00 | 652.50 | 658.68 | 654.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 651.15 | 657.17 | 654.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:15:00 | 650.90 | 657.17 | 654.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 654.75 | 655.28 | 654.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 654.90 | 655.28 | 654.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 658.70 | 655.96 | 654.44 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 653.35 | 656.95 | 657.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 647.75 | 655.11 | 656.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 657.20 | 653.15 | 654.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 11:15:00 | 657.20 | 653.15 | 654.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 657.20 | 653.15 | 654.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 657.20 | 653.15 | 654.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 651.95 | 652.91 | 654.57 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 686.00 | 659.13 | 656.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 699.55 | 671.74 | 663.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 684.10 | 689.47 | 679.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 684.10 | 689.47 | 679.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 686.40 | 693.62 | 687.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 687.85 | 693.62 | 687.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 687.20 | 692.34 | 687.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 686.85 | 692.34 | 687.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 692.95 | 692.46 | 688.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 684.00 | 692.46 | 688.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 695.55 | 700.70 | 697.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 695.55 | 700.70 | 697.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 692.90 | 699.14 | 696.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 692.90 | 699.14 | 696.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 690.95 | 696.37 | 695.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 690.95 | 696.37 | 695.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 700.00 | 696.77 | 696.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 690.90 | 696.77 | 696.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 683.45 | 694.11 | 694.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 677.95 | 690.87 | 693.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 687.60 | 682.00 | 686.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 687.60 | 682.00 | 686.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 687.60 | 682.00 | 686.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 687.60 | 682.00 | 686.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 687.40 | 683.08 | 686.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 678.30 | 683.08 | 686.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 690.30 | 681.99 | 684.70 | SL hit (close>static) qty=1.00 sl=688.85 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 686.80 | 681.04 | 680.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 693.95 | 689.16 | 685.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 15:15:00 | 691.00 | 691.24 | 688.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 15:15:00 | 691.00 | 691.24 | 688.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 691.00 | 691.24 | 688.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 693.70 | 691.16 | 688.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 684.80 | 690.62 | 690.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 684.80 | 690.62 | 690.84 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 696.40 | 691.31 | 690.95 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 686.75 | 690.54 | 690.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 679.80 | 688.29 | 689.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 667.30 | 663.51 | 670.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 667.30 | 663.51 | 670.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 667.30 | 663.51 | 670.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 668.85 | 663.51 | 670.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 668.60 | 664.52 | 670.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 671.15 | 664.52 | 670.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 665.00 | 664.15 | 668.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 665.00 | 664.15 | 668.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 666.15 | 661.16 | 664.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 666.15 | 661.16 | 664.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 666.15 | 662.16 | 664.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 663.20 | 663.10 | 664.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 673.95 | 666.21 | 666.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 673.95 | 666.21 | 666.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 677.10 | 670.09 | 668.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 675.15 | 675.37 | 672.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 675.15 | 675.37 | 672.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 675.15 | 675.37 | 672.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 672.10 | 675.37 | 672.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 676.50 | 677.62 | 674.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 698.25 | 679.88 | 676.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 709.80 | 713.68 | 713.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 709.80 | 713.68 | 713.84 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 720.90 | 712.92 | 712.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 725.55 | 715.45 | 713.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 14:15:00 | 728.80 | 732.70 | 725.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 15:00:00 | 728.80 | 732.70 | 725.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 709.80 | 727.21 | 724.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 708.30 | 727.21 | 724.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 709.90 | 723.75 | 722.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 709.90 | 723.75 | 722.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 712.50 | 721.50 | 722.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 708.50 | 714.75 | 717.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 719.00 | 712.89 | 715.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 719.00 | 712.89 | 715.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 719.00 | 712.89 | 715.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 719.00 | 712.89 | 715.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 705.60 | 711.44 | 714.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 702.70 | 709.66 | 713.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 702.55 | 709.66 | 713.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 702.10 | 707.37 | 711.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 702.95 | 706.85 | 709.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 708.15 | 706.49 | 708.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 701.70 | 705.38 | 707.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 703.00 | 705.38 | 707.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:15:00 | 697.20 | 705.43 | 706.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 667.57 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 667.42 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 667.00 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 667.80 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 666.62 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 667.85 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 662.34 | 679.22 | 687.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 14:15:00 | 632.43 | 671.47 | 682.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 681.05 | 676.36 | 675.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 688.30 | 679.33 | 677.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 701.75 | 703.81 | 698.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 701.75 | 703.81 | 698.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 699.90 | 703.03 | 698.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 699.90 | 703.03 | 698.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 697.00 | 701.82 | 698.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 693.50 | 700.16 | 698.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 689.50 | 698.03 | 697.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 689.50 | 698.03 | 697.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 693.75 | 696.31 | 696.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 689.00 | 693.67 | 695.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 698.70 | 684.20 | 687.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 698.70 | 684.20 | 687.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 698.70 | 684.20 | 687.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 698.70 | 684.20 | 687.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 698.70 | 687.10 | 688.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 698.70 | 687.10 | 688.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 693.10 | 690.22 | 689.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 705.65 | 694.71 | 692.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 717.25 | 717.55 | 711.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 717.25 | 717.55 | 711.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 716.15 | 717.27 | 712.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 714.25 | 717.27 | 712.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 712.50 | 716.32 | 712.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 712.50 | 716.32 | 712.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 711.90 | 715.43 | 712.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 711.90 | 715.43 | 712.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 710.60 | 714.47 | 711.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 710.60 | 714.47 | 711.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 711.00 | 713.77 | 711.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 711.40 | 713.77 | 711.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 722.70 | 715.56 | 712.83 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 704.45 | 711.00 | 711.75 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 715.35 | 712.50 | 712.23 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 707.00 | 712.16 | 712.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 697.50 | 707.92 | 710.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 708.50 | 702.54 | 705.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 708.50 | 702.54 | 705.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 708.50 | 702.54 | 705.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 708.50 | 702.54 | 705.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 710.35 | 704.11 | 706.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 709.05 | 704.11 | 706.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 705.50 | 704.70 | 706.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 706.05 | 704.70 | 706.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 703.55 | 704.47 | 705.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 703.55 | 704.47 | 705.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 702.90 | 703.86 | 705.33 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 706.60 | 705.96 | 705.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 713.05 | 707.38 | 706.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 744.50 | 747.21 | 737.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 744.50 | 747.21 | 737.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 749.65 | 746.05 | 740.43 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 730.00 | 737.73 | 738.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 721.85 | 733.16 | 735.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 688.60 | 681.42 | 692.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 688.60 | 681.42 | 692.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 688.60 | 681.42 | 692.13 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 700.85 | 689.83 | 688.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 14:15:00 | 702.25 | 695.48 | 691.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 693.00 | 694.98 | 691.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 687.85 | 694.98 | 691.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 687.05 | 693.40 | 691.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 685.00 | 693.40 | 691.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 680.00 | 690.72 | 690.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 680.00 | 690.72 | 690.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 682.55 | 689.08 | 689.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 679.60 | 685.29 | 687.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 664.00 | 657.86 | 666.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 664.00 | 657.86 | 666.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 664.00 | 657.86 | 666.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 652.00 | 656.32 | 664.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 652.50 | 655.13 | 660.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 619.40 | 629.59 | 640.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 619.88 | 629.59 | 640.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 635.10 | 628.68 | 636.96 | SL hit (close>ema200) qty=0.50 sl=628.68 alert=retest2 |

### Cycle 122 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 635.00 | 632.71 | 632.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 15:15:00 | 638.05 | 634.35 | 633.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 635.25 | 635.69 | 634.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 635.25 | 635.69 | 634.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 624.20 | 633.39 | 633.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 622.30 | 633.39 | 633.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 625.65 | 631.84 | 632.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 620.00 | 628.41 | 630.89 | Break + close below crossover candle low |

### Cycle 124 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 664.75 | 627.23 | 627.11 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 621.05 | 645.09 | 646.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 616.50 | 629.47 | 636.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 630.55 | 621.28 | 627.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 630.55 | 621.28 | 627.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 630.55 | 621.28 | 627.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 630.55 | 621.28 | 627.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 654.10 | 627.84 | 630.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 656.60 | 627.84 | 630.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 663.70 | 635.02 | 633.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 667.35 | 641.48 | 636.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 663.15 | 666.84 | 661.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 663.15 | 666.84 | 661.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 663.15 | 666.84 | 661.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 674.90 | 666.70 | 662.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 674.65 | 667.21 | 662.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 648.45 | 662.78 | 662.21 | SL hit (close<static) qty=1.00 sl=660.20 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 648.60 | 659.94 | 660.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 642.05 | 652.70 | 656.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 636.30 | 633.59 | 641.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 638.00 | 633.59 | 641.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 634.40 | 633.60 | 640.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 639.00 | 633.60 | 640.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 607.15 | 605.81 | 613.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 613.00 | 605.81 | 613.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 601.55 | 600.50 | 605.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 598.30 | 600.73 | 604.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 597.80 | 600.46 | 603.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 597.10 | 600.46 | 603.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 568.38 | 585.04 | 589.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 567.91 | 585.04 | 589.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 567.25 | 585.04 | 589.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 561.80 | 560.03 | 566.09 | SL hit (close>ema200) qty=0.50 sl=560.03 alert=retest2 |

### Cycle 128 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 574.00 | 559.58 | 558.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 578.30 | 567.76 | 563.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 571.30 | 577.02 | 570.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 571.30 | 577.02 | 570.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 571.30 | 577.02 | 570.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 571.30 | 577.02 | 570.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 570.35 | 575.68 | 570.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 573.00 | 575.68 | 570.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 579.85 | 574.24 | 570.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 562.00 | 574.44 | 572.75 | SL hit (close<static) qty=1.00 sl=564.10 alert=retest2 |

### Cycle 129 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 556.10 | 570.77 | 571.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 553.90 | 565.18 | 568.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 565.20 | 565.19 | 568.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:45:00 | 566.50 | 565.19 | 568.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 557.80 | 553.23 | 557.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 557.80 | 553.23 | 557.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 557.60 | 554.10 | 557.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:30:00 | 557.70 | 554.10 | 557.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 557.70 | 554.82 | 557.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:15:00 | 555.75 | 554.82 | 557.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 562.60 | 556.38 | 557.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:45:00 | 561.40 | 556.38 | 557.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 564.35 | 557.97 | 558.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 564.35 | 557.97 | 558.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 565.10 | 559.40 | 559.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 576.50 | 562.82 | 560.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 570.35 | 573.66 | 568.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 570.35 | 573.66 | 568.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 570.35 | 573.66 | 568.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 570.35 | 573.66 | 568.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 569.40 | 572.80 | 568.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 571.00 | 572.80 | 568.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 558.60 | 576.96 | 575.30 | SL hit (close<static) qty=1.00 sl=567.20 alert=retest2 |

### Cycle 131 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 562.60 | 572.19 | 573.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 560.75 | 567.38 | 570.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 567.50 | 565.57 | 568.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 567.50 | 565.57 | 568.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 567.50 | 565.57 | 568.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 567.50 | 565.57 | 568.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 571.05 | 566.67 | 568.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 571.35 | 566.67 | 568.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 571.80 | 567.69 | 569.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 571.80 | 567.69 | 569.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 576.10 | 569.37 | 569.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 582.00 | 569.37 | 569.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 589.70 | 573.44 | 571.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 594.60 | 580.18 | 575.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 583.00 | 583.62 | 578.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 579.45 | 583.62 | 578.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 571.65 | 581.23 | 577.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 571.65 | 581.23 | 577.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 565.85 | 578.15 | 576.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 566.30 | 578.15 | 576.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 562.70 | 575.06 | 575.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 560.70 | 572.19 | 574.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 568.80 | 555.42 | 560.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 568.80 | 555.42 | 560.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 568.80 | 555.42 | 560.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 569.80 | 555.42 | 560.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 565.15 | 557.36 | 561.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 566.85 | 557.36 | 561.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 572.75 | 565.08 | 564.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 579.90 | 568.94 | 566.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 560.30 | 567.21 | 565.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 560.30 | 567.21 | 565.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 560.30 | 567.21 | 565.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 557.00 | 567.21 | 565.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 563.65 | 566.50 | 565.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 565.70 | 566.54 | 565.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 566.05 | 572.00 | 569.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 622.27 | 604.74 | 593.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 720.85 | 724.86 | 725.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 715.10 | 721.35 | 723.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 710.80 | 707.48 | 712.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 710.80 | 707.48 | 712.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 710.80 | 707.48 | 712.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 713.05 | 707.48 | 712.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 711.35 | 708.25 | 712.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:15:00 | 706.10 | 708.25 | 712.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 706.10 | 707.82 | 711.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 726.95 | 707.82 | 711.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 724.70 | 711.20 | 713.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 728.50 | 711.20 | 713.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 718.55 | 714.64 | 714.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 726.05 | 720.99 | 717.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 725.25 | 725.74 | 722.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 725.10 | 725.74 | 722.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 742.40 | 749.29 | 744.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 742.40 | 749.29 | 744.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 743.40 | 748.11 | 744.51 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-22 11:30:00 | 518.80 | 2024-05-30 14:15:00 | 492.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-22 13:45:00 | 518.10 | 2024-05-30 14:15:00 | 492.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 10:15:00 | 517.95 | 2024-05-30 14:15:00 | 492.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-22 11:30:00 | 518.80 | 2024-05-30 15:15:00 | 506.00 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2024-05-22 13:45:00 | 518.10 | 2024-05-30 15:15:00 | 506.00 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2024-05-23 10:15:00 | 517.95 | 2024-05-30 15:15:00 | 506.00 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2024-05-23 14:30:00 | 516.60 | 2024-06-03 13:15:00 | 506.40 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2024-05-24 12:30:00 | 513.40 | 2024-06-03 13:15:00 | 506.40 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2024-06-10 09:15:00 | 509.45 | 2024-06-14 11:15:00 | 512.75 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-06-25 14:00:00 | 529.95 | 2024-06-26 15:15:00 | 535.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-06-26 09:15:00 | 529.85 | 2024-06-26 15:15:00 | 535.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-06-26 09:45:00 | 529.25 | 2024-06-26 15:15:00 | 535.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-06-26 11:15:00 | 530.00 | 2024-06-26 15:15:00 | 535.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-07-05 15:15:00 | 537.80 | 2024-07-08 10:15:00 | 529.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-07-09 09:15:00 | 537.90 | 2024-07-09 12:15:00 | 591.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-07-29 09:15:00 | 608.05 | 2024-07-31 09:15:00 | 604.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-01 09:45:00 | 614.30 | 2024-08-05 13:15:00 | 614.15 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-08-05 09:30:00 | 614.55 | 2024-08-05 13:15:00 | 614.15 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-08-05 13:15:00 | 610.85 | 2024-08-05 13:15:00 | 614.15 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest1 | 2024-08-16 09:15:00 | 647.35 | 2024-08-19 09:15:00 | 661.05 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest1 | 2024-08-16 11:30:00 | 649.00 | 2024-08-19 09:15:00 | 661.05 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-08-19 11:45:00 | 643.45 | 2024-08-22 10:15:00 | 662.45 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-08-19 13:00:00 | 645.55 | 2024-08-22 10:15:00 | 662.45 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-09-16 12:00:00 | 772.90 | 2024-09-19 15:15:00 | 759.95 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-10 13:15:00 | 734.00 | 2024-10-15 12:15:00 | 740.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-10-10 14:30:00 | 733.60 | 2024-10-15 12:15:00 | 740.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-11 10:15:00 | 732.50 | 2024-10-15 12:15:00 | 740.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-11 13:00:00 | 733.00 | 2024-10-15 12:15:00 | 740.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-14 10:15:00 | 720.00 | 2024-10-15 12:15:00 | 740.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-10-25 09:45:00 | 666.85 | 2024-10-30 09:15:00 | 701.75 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2024-10-25 13:30:00 | 667.95 | 2024-10-30 09:15:00 | 701.75 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-10-29 10:30:00 | 664.00 | 2024-10-30 09:15:00 | 701.75 | STOP_HIT | 1.00 | -5.69% |
| SELL | retest2 | 2024-10-29 13:15:00 | 666.15 | 2024-10-30 09:15:00 | 701.75 | STOP_HIT | 1.00 | -5.34% |
| BUY | retest2 | 2024-11-04 12:15:00 | 720.05 | 2024-11-08 12:15:00 | 721.85 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-11-05 09:45:00 | 722.20 | 2024-11-08 12:15:00 | 721.85 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-11-05 12:15:00 | 720.75 | 2024-11-08 12:15:00 | 721.85 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-11-05 13:00:00 | 723.95 | 2024-11-08 12:15:00 | 721.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-11-29 09:45:00 | 752.90 | 2024-12-06 11:15:00 | 782.35 | STOP_HIT | 1.00 | 3.91% |
| BUY | retest1 | 2025-01-01 09:15:00 | 830.45 | 2025-01-01 10:15:00 | 816.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-01-02 09:15:00 | 839.65 | 2025-01-03 11:15:00 | 814.95 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-01-02 10:30:00 | 834.00 | 2025-01-03 11:15:00 | 814.95 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-01-20 09:45:00 | 701.45 | 2025-01-22 09:15:00 | 666.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 11:45:00 | 698.80 | 2025-01-22 09:15:00 | 663.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 695.45 | 2025-01-22 09:15:00 | 660.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 09:45:00 | 701.45 | 2025-01-23 09:15:00 | 689.55 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2025-01-20 11:45:00 | 698.80 | 2025-01-23 09:15:00 | 689.55 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-01-21 10:15:00 | 695.45 | 2025-01-23 09:15:00 | 689.55 | STOP_HIT | 0.50 | 0.85% |
| BUY | retest2 | 2025-01-31 15:00:00 | 699.45 | 2025-02-10 10:15:00 | 714.35 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-02-01 09:15:00 | 701.30 | 2025-02-10 10:15:00 | 714.35 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-02-01 10:30:00 | 699.55 | 2025-02-10 10:15:00 | 714.35 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-02-01 13:45:00 | 699.05 | 2025-02-10 10:15:00 | 714.35 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-02-18 11:30:00 | 633.10 | 2025-02-19 09:15:00 | 667.40 | STOP_HIT | 1.00 | -5.42% |
| BUY | retest2 | 2025-03-12 14:00:00 | 675.45 | 2025-03-17 15:15:00 | 669.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-03-12 14:45:00 | 675.95 | 2025-03-17 15:15:00 | 669.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-03-13 10:00:00 | 677.85 | 2025-03-17 15:15:00 | 669.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-03-13 13:45:00 | 681.45 | 2025-03-17 15:15:00 | 669.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-03-27 14:00:00 | 660.35 | 2025-04-03 10:15:00 | 672.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-03-28 11:15:00 | 657.10 | 2025-04-03 10:15:00 | 672.30 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-04-09 09:15:00 | 601.15 | 2025-04-11 12:15:00 | 632.20 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2025-04-17 13:30:00 | 659.85 | 2025-04-25 09:15:00 | 670.20 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-04-17 14:15:00 | 660.40 | 2025-04-25 09:15:00 | 670.20 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-04-21 09:15:00 | 661.80 | 2025-04-25 09:15:00 | 670.20 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-05-08 10:45:00 | 699.20 | 2025-05-08 15:15:00 | 671.10 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2025-05-13 14:15:00 | 728.00 | 2025-05-13 14:15:00 | 702.00 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-06-04 12:00:00 | 726.40 | 2025-06-05 15:15:00 | 713.20 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-10 11:00:00 | 701.80 | 2025-06-11 14:15:00 | 666.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 11:00:00 | 701.80 | 2025-06-12 09:15:00 | 691.05 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-06-19 12:45:00 | 709.40 | 2025-06-23 14:15:00 | 723.90 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-06-19 14:30:00 | 711.00 | 2025-06-23 14:15:00 | 723.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-06-20 11:30:00 | 711.80 | 2025-06-23 14:15:00 | 723.90 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-30 09:15:00 | 769.25 | 2025-07-01 09:15:00 | 746.15 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest1 | 2025-07-07 11:30:00 | 792.60 | 2025-07-11 10:15:00 | 806.10 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-07-10 09:15:00 | 813.15 | 2025-07-22 09:15:00 | 825.00 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-07-11 12:30:00 | 806.65 | 2025-07-22 09:15:00 | 825.00 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-07-14 09:30:00 | 807.90 | 2025-07-22 09:15:00 | 825.00 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-07-31 13:15:00 | 817.65 | 2025-07-31 14:15:00 | 785.00 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-08-18 14:30:00 | 707.60 | 2025-08-20 13:15:00 | 703.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-08-18 15:15:00 | 711.00 | 2025-08-20 13:15:00 | 703.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-28 10:45:00 | 731.00 | 2025-09-05 10:15:00 | 724.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-28 14:30:00 | 728.65 | 2025-09-05 10:15:00 | 724.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-28 15:00:00 | 729.30 | 2025-09-05 10:15:00 | 724.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-08-29 09:45:00 | 731.00 | 2025-09-05 10:15:00 | 724.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-09 11:30:00 | 718.20 | 2025-09-16 12:15:00 | 717.90 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-09-10 13:45:00 | 718.00 | 2025-09-16 12:15:00 | 717.90 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-09-11 10:45:00 | 716.70 | 2025-09-16 12:15:00 | 717.90 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-09-26 09:15:00 | 655.15 | 2025-10-03 09:15:00 | 662.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-23 09:15:00 | 678.30 | 2025-10-23 12:15:00 | 690.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-10-23 15:15:00 | 680.10 | 2025-10-28 10:15:00 | 686.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-24 09:30:00 | 680.95 | 2025-10-28 10:15:00 | 686.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-28 10:15:00 | 682.65 | 2025-10-28 10:15:00 | 686.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-30 11:00:00 | 693.70 | 2025-10-31 13:15:00 | 684.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-12 09:15:00 | 663.20 | 2025-11-12 10:15:00 | 673.95 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-17 09:15:00 | 698.25 | 2025-11-24 11:15:00 | 709.80 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-01 11:30:00 | 702.70 | 2025-12-08 12:15:00 | 667.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 12:00:00 | 702.55 | 2025-12-08 12:15:00 | 667.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 702.10 | 2025-12-08 12:15:00 | 667.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 15:00:00 | 702.95 | 2025-12-08 12:15:00 | 667.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 14:45:00 | 701.70 | 2025-12-08 12:15:00 | 666.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 15:15:00 | 703.00 | 2025-12-08 12:15:00 | 667.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:15:00 | 697.20 | 2025-12-08 12:15:00 | 662.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:30:00 | 702.70 | 2025-12-08 14:15:00 | 632.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 12:00:00 | 702.55 | 2025-12-08 14:15:00 | 632.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 702.10 | 2025-12-08 14:15:00 | 631.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 15:00:00 | 702.95 | 2025-12-08 14:15:00 | 632.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 14:45:00 | 701.70 | 2025-12-08 14:15:00 | 631.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 15:15:00 | 703.00 | 2025-12-08 14:15:00 | 632.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-04 11:15:00 | 697.20 | 2025-12-08 14:15:00 | 627.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 652.00 | 2026-01-27 13:15:00 | 619.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 652.50 | 2026-01-27 13:15:00 | 619.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 652.00 | 2026-01-28 09:15:00 | 635.10 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2026-01-23 09:15:00 | 652.50 | 2026-01-28 09:15:00 | 635.10 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest2 | 2026-02-12 12:15:00 | 674.90 | 2026-02-13 09:15:00 | 648.45 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-02-12 13:15:00 | 674.65 | 2026-02-13 09:15:00 | 648.45 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2026-02-25 09:30:00 | 598.30 | 2026-03-02 09:15:00 | 568.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 597.80 | 2026-03-02 09:15:00 | 567.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 597.10 | 2026-03-02 09:15:00 | 567.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 598.30 | 2026-03-05 12:15:00 | 561.80 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2026-02-25 10:30:00 | 597.80 | 2026-03-05 12:15:00 | 561.80 | STOP_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2026-02-25 11:15:00 | 597.10 | 2026-03-05 12:15:00 | 561.80 | STOP_HIT | 0.50 | 5.91% |
| BUY | retest2 | 2026-03-12 09:15:00 | 573.00 | 2026-03-13 09:15:00 | 562.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-03-12 12:00:00 | 579.85 | 2026-03-13 09:15:00 | 562.00 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-03-19 11:15:00 | 571.00 | 2026-03-23 09:15:00 | 558.60 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-04-02 11:30:00 | 565.70 | 2026-04-08 09:15:00 | 622.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 566.05 | 2026-04-08 09:15:00 | 622.65 | TARGET_HIT | 1.00 | 10.00% |
