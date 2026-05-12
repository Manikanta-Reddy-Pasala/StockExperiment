# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 515.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 102 |
| ALERT2 | 100 |
| ALERT2_SKIP | 53 |
| ALERT3 | 286 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 116 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 32 / 90
- **Target hits / Stop hits / Partials:** 3 / 113 / 6
- **Avg / median % per leg:** 0.13% / -0.77%
- **Sum % (uncompounded):** 15.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 11 | 22.0% | 3 | 46 | 1 | 0.19% | 9.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.25% | 10.5% |
| BUY @ 3rd Alert (retest2) | 48 | 9 | 18.8% | 3 | 45 | 0 | -0.02% | -0.9% |
| SELL (all) | 72 | 21 | 29.2% | 0 | 67 | 5 | 0.08% | 5.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 21 | 29.2% | 0 | 67 | 5 | 0.08% | 5.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.25% | 10.5% |
| retest2 (combined) | 120 | 30 | 25.0% | 3 | 112 | 5 | 0.04% | 5.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 494.50 | 494.14 | 494.11 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 481.35 | 491.99 | 493.17 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 491.45 | 489.21 | 489.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 493.40 | 490.05 | 489.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 486.85 | 490.95 | 490.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 486.85 | 490.95 | 490.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 486.85 | 490.95 | 490.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 484.95 | 490.95 | 490.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 487.65 | 490.29 | 490.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 485.20 | 490.29 | 490.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 488.70 | 489.95 | 489.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:45:00 | 489.25 | 489.95 | 489.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 492.10 | 490.38 | 490.13 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 488.00 | 489.98 | 490.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 14:15:00 | 487.70 | 489.53 | 489.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 491.45 | 489.82 | 489.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 491.45 | 489.82 | 489.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 491.45 | 489.82 | 489.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 491.45 | 489.82 | 489.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 489.40 | 489.74 | 489.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:15:00 | 488.35 | 489.39 | 489.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 488.20 | 488.14 | 488.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 488.50 | 487.63 | 488.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 490.75 | 488.54 | 488.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 490.75 | 488.54 | 488.46 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 486.35 | 488.10 | 488.27 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 491.80 | 488.92 | 488.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 11:15:00 | 493.40 | 489.82 | 489.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 493.25 | 495.75 | 493.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 493.25 | 495.75 | 493.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 493.25 | 495.75 | 493.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 493.25 | 495.75 | 493.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 490.95 | 494.79 | 493.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 488.35 | 494.79 | 493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 486.90 | 493.21 | 492.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 487.70 | 493.21 | 492.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 484.40 | 491.45 | 491.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 12:15:00 | 483.40 | 488.81 | 490.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 10:15:00 | 471.50 | 470.03 | 477.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 11:00:00 | 471.50 | 470.03 | 477.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 472.75 | 462.73 | 466.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 472.75 | 462.73 | 466.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 476.50 | 465.48 | 467.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 477.90 | 465.48 | 467.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 480.30 | 470.88 | 470.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 481.75 | 477.70 | 475.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 491.25 | 492.31 | 488.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:00:00 | 491.25 | 492.31 | 488.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 500.00 | 503.66 | 502.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 500.00 | 503.66 | 502.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 500.90 | 503.11 | 502.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 499.35 | 503.11 | 502.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 500.90 | 502.55 | 501.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 501.30 | 502.55 | 501.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 499.15 | 501.87 | 501.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 499.10 | 501.87 | 501.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 497.90 | 501.07 | 501.35 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 506.55 | 501.76 | 501.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 508.40 | 503.09 | 502.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 507.75 | 508.53 | 505.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 507.75 | 508.53 | 505.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 506.30 | 508.09 | 505.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:00:00 | 506.30 | 508.09 | 505.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 506.45 | 507.76 | 505.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 506.75 | 507.76 | 505.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 504.05 | 507.02 | 505.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 504.05 | 507.02 | 505.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 505.45 | 506.70 | 505.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 504.30 | 506.70 | 505.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 502.05 | 505.77 | 505.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 501.00 | 505.77 | 505.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 503.40 | 505.30 | 505.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 500.55 | 505.30 | 505.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 500.75 | 504.39 | 504.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 495.45 | 499.19 | 500.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 500.15 | 498.87 | 500.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 500.15 | 498.87 | 500.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 500.15 | 498.87 | 500.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 500.15 | 498.87 | 500.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 499.90 | 499.08 | 500.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 497.70 | 498.85 | 499.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 501.20 | 499.32 | 500.06 | SL hit (close>static) qty=1.00 sl=501.15 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 503.50 | 500.50 | 500.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 504.50 | 502.75 | 501.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 506.50 | 506.97 | 505.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 506.50 | 506.97 | 505.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 506.50 | 506.97 | 505.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 506.50 | 506.97 | 505.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 511.00 | 511.00 | 509.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 515.45 | 511.47 | 509.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 514.00 | 512.71 | 510.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:00:00 | 514.40 | 513.05 | 510.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 514.90 | 513.19 | 511.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 518.95 | 514.34 | 512.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 515.70 | 514.34 | 512.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 514.05 | 514.78 | 513.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 512.45 | 514.78 | 513.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 514.00 | 514.62 | 513.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 513.50 | 514.62 | 513.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 514.00 | 514.50 | 513.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 514.00 | 514.50 | 513.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 514.55 | 514.51 | 513.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 514.55 | 514.51 | 513.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 510.65 | 513.74 | 513.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 510.65 | 513.74 | 513.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 510.05 | 513.00 | 512.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 509.70 | 513.00 | 512.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-08 13:15:00 | 509.60 | 512.32 | 512.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 509.60 | 512.32 | 512.62 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 513.65 | 512.67 | 512.54 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 511.00 | 512.48 | 512.57 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 13:15:00 | 524.05 | 514.82 | 513.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 526.70 | 517.20 | 514.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 11:15:00 | 525.95 | 526.45 | 523.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 12:00:00 | 525.95 | 526.45 | 523.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 522.65 | 525.69 | 523.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 522.65 | 525.69 | 523.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 522.75 | 525.10 | 523.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 523.25 | 525.10 | 523.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 519.75 | 524.03 | 523.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 519.75 | 524.03 | 523.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 520.45 | 523.32 | 523.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 521.35 | 523.32 | 523.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 520.05 | 522.34 | 522.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 520.05 | 522.34 | 522.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 518.40 | 520.74 | 521.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 11:15:00 | 521.90 | 520.97 | 521.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 11:15:00 | 521.90 | 520.97 | 521.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 521.90 | 520.97 | 521.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:00:00 | 521.90 | 520.97 | 521.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 516.20 | 520.01 | 521.15 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 10:15:00 | 524.85 | 522.14 | 521.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 529.75 | 526.01 | 525.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 528.90 | 529.41 | 527.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:00:00 | 528.90 | 529.41 | 527.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 533.50 | 530.23 | 528.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 528.60 | 530.23 | 528.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 530.05 | 530.25 | 528.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:30:00 | 529.30 | 530.25 | 528.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 531.00 | 530.40 | 528.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 535.00 | 530.40 | 528.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 546.65 | 550.31 | 550.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 546.65 | 550.31 | 550.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 536.40 | 547.53 | 549.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 546.60 | 545.10 | 547.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 546.60 | 545.10 | 547.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 546.60 | 545.10 | 547.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 546.60 | 545.10 | 547.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 544.70 | 545.02 | 546.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 540.65 | 543.29 | 545.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 539.90 | 543.29 | 545.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 535.75 | 529.32 | 528.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 535.75 | 529.32 | 528.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 11:15:00 | 539.60 | 534.28 | 532.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 09:15:00 | 538.35 | 540.86 | 536.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:00:00 | 538.35 | 540.86 | 536.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 578.05 | 567.78 | 560.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:15:00 | 580.40 | 567.78 | 560.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 595.50 | 573.92 | 567.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 565.60 | 572.20 | 572.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 565.60 | 572.20 | 572.89 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 575.75 | 571.44 | 571.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 580.15 | 573.18 | 572.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 577.85 | 580.21 | 576.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 577.85 | 580.21 | 576.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 577.85 | 580.21 | 576.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 577.25 | 580.21 | 576.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 578.00 | 579.77 | 577.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 576.90 | 579.77 | 577.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 575.45 | 578.91 | 576.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:00:00 | 575.45 | 578.91 | 576.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 575.05 | 578.13 | 576.74 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 573.50 | 575.70 | 575.89 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 575.75 | 573.61 | 573.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 580.05 | 575.46 | 574.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 576.10 | 576.89 | 575.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 576.10 | 576.89 | 575.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 588.00 | 579.24 | 577.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:15:00 | 593.20 | 583.28 | 579.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:15:00 | 592.15 | 593.88 | 590.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 615.00 | 619.80 | 619.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 615.00 | 619.80 | 619.91 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 623.60 | 619.85 | 619.59 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 09:15:00 | 616.60 | 619.95 | 619.97 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 622.05 | 620.05 | 619.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 624.55 | 620.95 | 620.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 621.00 | 621.80 | 621.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 621.00 | 621.80 | 621.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 621.00 | 621.80 | 621.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 619.25 | 621.80 | 621.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 613.80 | 620.20 | 620.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 613.50 | 615.80 | 617.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 13:15:00 | 609.60 | 609.11 | 612.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 14:00:00 | 609.60 | 609.11 | 612.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 616.10 | 610.51 | 612.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 616.10 | 610.51 | 612.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 615.35 | 611.48 | 613.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 611.90 | 611.48 | 613.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 619.15 | 612.90 | 612.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 619.15 | 612.90 | 612.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 624.00 | 615.12 | 613.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 618.95 | 619.93 | 616.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 618.95 | 619.93 | 616.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 618.95 | 619.93 | 616.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 618.95 | 619.93 | 616.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 618.10 | 619.57 | 617.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 617.50 | 619.57 | 617.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 623.45 | 620.34 | 617.62 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 611.50 | 617.62 | 617.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 604.40 | 614.51 | 616.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 573.60 | 571.77 | 580.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 573.60 | 571.77 | 580.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 577.00 | 573.62 | 577.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:15:00 | 574.80 | 574.22 | 577.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 574.90 | 576.36 | 577.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:45:00 | 574.85 | 576.18 | 577.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 14:30:00 | 574.65 | 575.55 | 576.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 566.30 | 570.77 | 572.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 584.70 | 574.72 | 573.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 584.70 | 574.72 | 573.40 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 570.25 | 574.62 | 575.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 566.35 | 571.60 | 573.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 556.25 | 556.00 | 561.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:45:00 | 556.40 | 556.00 | 561.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 559.15 | 556.79 | 560.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 555.70 | 556.37 | 559.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 553.50 | 556.93 | 559.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 545.20 | 541.81 | 541.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 545.20 | 541.81 | 541.78 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 539.30 | 541.72 | 541.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 533.30 | 539.76 | 540.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 11:15:00 | 540.05 | 538.82 | 540.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 11:15:00 | 540.05 | 538.82 | 540.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 540.05 | 538.82 | 540.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 540.05 | 538.82 | 540.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 544.80 | 540.02 | 540.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 544.80 | 540.02 | 540.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 546.70 | 541.35 | 541.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 549.00 | 544.17 | 542.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 545.45 | 545.51 | 544.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 545.45 | 545.51 | 544.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 545.45 | 545.51 | 544.04 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 534.25 | 542.44 | 542.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 533.75 | 540.71 | 542.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 518.65 | 517.19 | 523.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 518.65 | 517.19 | 523.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 518.65 | 517.19 | 523.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:30:00 | 507.80 | 514.29 | 517.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 14:15:00 | 482.41 | 485.84 | 489.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 472.00 | 470.38 | 475.55 | SL hit (close>ema200) qty=0.50 sl=470.38 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 482.45 | 476.50 | 476.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 10:15:00 | 485.45 | 479.66 | 478.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 487.80 | 488.57 | 485.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 488.00 | 488.57 | 485.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 486.00 | 488.06 | 485.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 486.00 | 488.06 | 485.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 487.00 | 487.84 | 485.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 487.20 | 487.84 | 485.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 490.00 | 493.20 | 491.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 490.00 | 493.20 | 491.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 488.15 | 492.19 | 490.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 488.15 | 492.19 | 490.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 489.20 | 491.59 | 490.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 489.50 | 491.59 | 490.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 491.40 | 491.56 | 490.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 488.55 | 491.56 | 490.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 488.30 | 490.90 | 490.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 488.30 | 490.90 | 490.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 486.10 | 489.94 | 490.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 12:15:00 | 484.70 | 488.26 | 489.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 482.10 | 480.45 | 482.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 482.10 | 480.45 | 482.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 483.00 | 480.96 | 482.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:30:00 | 481.70 | 480.96 | 482.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 483.35 | 481.44 | 482.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 483.35 | 481.44 | 482.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 483.50 | 481.85 | 482.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 483.75 | 481.85 | 482.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 483.80 | 482.24 | 483.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 483.50 | 482.24 | 483.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 475.40 | 480.04 | 481.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 472.75 | 476.19 | 477.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 475.90 | 474.27 | 474.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 475.90 | 474.27 | 474.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 477.50 | 474.91 | 474.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 473.95 | 475.73 | 475.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 473.95 | 475.73 | 475.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 473.95 | 475.73 | 475.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 473.95 | 475.73 | 475.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 472.80 | 475.15 | 474.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 473.05 | 475.15 | 474.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 467.55 | 473.63 | 474.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 466.95 | 472.29 | 473.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 444.50 | 444.22 | 447.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:00:00 | 444.50 | 444.22 | 447.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 444.95 | 444.43 | 447.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 441.90 | 444.02 | 445.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:45:00 | 442.00 | 443.53 | 445.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:00:00 | 441.60 | 443.04 | 444.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 442.00 | 442.44 | 443.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 443.80 | 442.71 | 443.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 443.80 | 442.71 | 443.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 443.25 | 442.82 | 443.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 443.25 | 442.82 | 443.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 442.05 | 442.67 | 443.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 438.75 | 442.59 | 443.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 444.65 | 442.62 | 443.12 | SL hit (close>static) qty=1.00 sl=443.60 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 446.40 | 443.92 | 443.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 447.65 | 444.88 | 444.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 11:15:00 | 445.40 | 446.11 | 445.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 11:15:00 | 445.40 | 446.11 | 445.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 445.40 | 446.11 | 445.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 445.40 | 446.11 | 445.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 445.70 | 446.03 | 445.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 445.70 | 446.03 | 445.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 446.95 | 446.21 | 445.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 448.70 | 446.74 | 445.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 453.55 | 446.99 | 445.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 446.70 | 454.71 | 454.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 446.70 | 454.71 | 454.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 445.80 | 452.93 | 454.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 450.80 | 449.68 | 451.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 450.80 | 449.68 | 451.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 450.80 | 449.68 | 451.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 451.95 | 449.68 | 451.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 450.30 | 449.80 | 451.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 451.45 | 449.80 | 451.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 449.30 | 449.70 | 451.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 451.60 | 449.70 | 451.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 451.45 | 449.84 | 451.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 449.65 | 449.84 | 451.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 451.50 | 450.17 | 451.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 449.30 | 450.44 | 451.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 455.35 | 451.42 | 451.59 | SL hit (close>static) qty=1.00 sl=452.20 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 456.25 | 452.39 | 452.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 461.80 | 455.04 | 453.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 461.50 | 465.83 | 461.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 461.50 | 465.83 | 461.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 461.50 | 465.83 | 461.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 461.50 | 465.83 | 461.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 465.15 | 465.70 | 462.24 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 456.60 | 461.03 | 461.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 454.75 | 459.77 | 460.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 459.55 | 455.70 | 457.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 459.55 | 455.70 | 457.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 459.55 | 455.70 | 457.85 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 462.95 | 459.54 | 459.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 465.15 | 463.99 | 462.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 466.45 | 467.61 | 465.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 466.45 | 467.61 | 465.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 466.45 | 467.61 | 465.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 465.90 | 467.61 | 465.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 471.80 | 473.06 | 471.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 471.80 | 473.06 | 471.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 473.40 | 473.13 | 471.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:45:00 | 471.10 | 473.13 | 471.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 472.45 | 472.99 | 471.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:15:00 | 474.45 | 472.99 | 471.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 472.00 | 478.31 | 478.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 10:15:00 | 472.00 | 478.31 | 478.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 09:15:00 | 464.85 | 474.05 | 476.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 471.45 | 465.78 | 470.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 471.45 | 465.78 | 470.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 471.45 | 465.78 | 470.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 471.45 | 465.78 | 470.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 470.45 | 466.71 | 470.05 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 476.75 | 471.92 | 471.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 12:15:00 | 490.85 | 476.06 | 473.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 483.40 | 484.86 | 480.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 12:00:00 | 483.40 | 484.86 | 480.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 473.35 | 482.50 | 480.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 472.05 | 482.50 | 480.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 472.55 | 480.51 | 480.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 472.70 | 480.51 | 480.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 11:15:00 | 473.90 | 479.19 | 479.62 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 493.65 | 482.13 | 480.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 477.85 | 480.81 | 480.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 475.40 | 479.11 | 479.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 478.00 | 477.89 | 479.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 478.00 | 477.89 | 479.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 477.65 | 477.84 | 478.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 478.60 | 477.84 | 478.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 478.10 | 477.89 | 478.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 479.90 | 477.89 | 478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 476.65 | 477.64 | 478.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:15:00 | 477.55 | 477.64 | 478.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 479.40 | 477.99 | 478.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 479.40 | 477.99 | 478.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 480.50 | 478.50 | 478.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 479.90 | 478.50 | 478.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 480.30 | 478.86 | 478.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 480.30 | 478.86 | 478.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 10:15:00 | 480.25 | 479.13 | 479.10 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 477.00 | 478.71 | 478.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 475.60 | 478.09 | 478.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 13:15:00 | 479.30 | 478.33 | 478.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 13:15:00 | 479.30 | 478.33 | 478.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 479.30 | 478.33 | 478.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:00:00 | 479.30 | 478.33 | 478.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 14:15:00 | 484.05 | 479.47 | 479.16 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 473.60 | 478.44 | 478.94 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 11:15:00 | 483.80 | 479.20 | 478.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 12:15:00 | 489.35 | 481.23 | 479.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 486.00 | 486.07 | 483.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 486.00 | 486.07 | 483.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 481.70 | 484.98 | 483.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 481.70 | 484.98 | 483.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 481.70 | 484.33 | 483.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:45:00 | 481.80 | 484.33 | 483.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 476.20 | 481.80 | 482.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 474.80 | 480.40 | 481.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 11:15:00 | 479.80 | 479.39 | 480.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 11:15:00 | 479.80 | 479.39 | 480.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 479.80 | 479.39 | 480.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 479.80 | 479.39 | 480.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 481.90 | 479.89 | 480.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 481.90 | 479.89 | 480.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 481.65 | 480.24 | 480.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 484.25 | 480.24 | 480.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 483.55 | 481.57 | 481.38 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 479.00 | 481.06 | 481.17 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 482.10 | 481.26 | 481.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 489.25 | 482.86 | 481.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 488.45 | 488.66 | 486.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 12:30:00 | 488.15 | 488.66 | 486.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 489.50 | 488.74 | 486.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 10:00:00 | 490.45 | 489.12 | 487.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:45:00 | 490.15 | 489.43 | 487.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:00:00 | 489.90 | 489.34 | 487.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:30:00 | 491.85 | 489.98 | 488.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 484.90 | 489.27 | 488.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 484.90 | 489.27 | 488.36 | SL hit (close<static) qty=1.00 sl=485.45 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 490.30 | 498.80 | 499.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 488.40 | 495.39 | 497.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 489.35 | 488.09 | 491.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 489.35 | 488.09 | 491.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 481.10 | 482.62 | 484.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 489.10 | 482.62 | 484.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 495.00 | 485.10 | 485.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 495.00 | 485.10 | 485.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 497.95 | 487.67 | 486.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 506.75 | 497.90 | 492.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 507.00 | 507.51 | 503.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 13:45:00 | 507.35 | 507.51 | 503.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 500.95 | 505.81 | 502.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 499.05 | 505.81 | 502.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 500.00 | 504.65 | 502.71 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 498.85 | 501.48 | 501.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 488.60 | 497.86 | 499.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 10:15:00 | 486.00 | 485.45 | 490.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 11:00:00 | 486.00 | 485.45 | 490.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 489.15 | 486.55 | 490.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 489.15 | 486.55 | 490.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 497.40 | 488.96 | 490.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:00:00 | 497.40 | 488.96 | 490.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 500.15 | 491.19 | 491.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 500.15 | 491.19 | 491.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 502.25 | 493.41 | 492.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 502.70 | 497.62 | 494.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 09:15:00 | 498.50 | 498.82 | 495.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:45:00 | 496.40 | 498.82 | 495.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 501.50 | 503.41 | 501.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:30:00 | 501.00 | 503.41 | 501.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 500.40 | 502.81 | 501.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:15:00 | 502.15 | 502.81 | 501.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 499.90 | 502.23 | 501.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 499.90 | 502.23 | 501.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 504.45 | 502.67 | 501.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 500.10 | 502.67 | 501.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 501.65 | 504.70 | 503.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:00:00 | 501.65 | 504.70 | 503.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 500.40 | 503.84 | 503.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:00:00 | 500.40 | 503.84 | 503.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 12:15:00 | 499.60 | 502.38 | 502.49 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 503.25 | 502.38 | 502.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 12:15:00 | 506.15 | 503.26 | 502.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 506.45 | 507.22 | 505.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:00:00 | 506.45 | 507.22 | 505.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 503.55 | 506.48 | 505.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 502.75 | 506.48 | 505.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 505.50 | 506.29 | 505.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 506.65 | 506.22 | 505.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 501.30 | 505.24 | 504.89 | SL hit (close<static) qty=1.00 sl=503.55 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 500.95 | 504.38 | 504.53 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 509.00 | 505.30 | 504.94 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 503.00 | 504.62 | 504.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 500.50 | 502.98 | 503.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 502.85 | 501.51 | 502.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 502.85 | 501.51 | 502.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 502.85 | 501.51 | 502.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 502.85 | 501.51 | 502.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 504.90 | 502.19 | 502.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 508.20 | 502.19 | 502.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 507.40 | 503.23 | 503.29 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 508.80 | 504.34 | 503.79 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 500.25 | 503.28 | 503.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 493.75 | 499.93 | 501.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 501.30 | 497.30 | 499.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 501.30 | 497.30 | 499.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 501.30 | 497.30 | 499.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 500.55 | 497.30 | 499.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 506.10 | 499.06 | 499.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 506.10 | 499.06 | 499.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 503.85 | 500.59 | 500.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 506.80 | 503.11 | 501.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 508.10 | 508.44 | 505.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 508.10 | 508.44 | 505.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 498.60 | 509.11 | 507.78 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 502.40 | 506.82 | 506.91 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 12:15:00 | 508.20 | 507.10 | 507.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 14:15:00 | 511.50 | 508.12 | 507.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 09:15:00 | 531.00 | 534.71 | 528.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 10:00:00 | 531.00 | 534.71 | 528.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 528.80 | 533.53 | 528.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:30:00 | 526.75 | 533.53 | 528.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 530.50 | 532.92 | 528.89 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 10:15:00 | 521.65 | 527.92 | 528.05 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 545.80 | 530.49 | 528.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 546.80 | 540.60 | 535.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 543.10 | 543.34 | 539.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 553.70 | 543.34 | 539.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 14:15:00 | 581.39 | 569.78 | 559.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-24 11:15:00 | 584.10 | 584.53 | 575.59 | SL hit (close<ema200) qty=0.50 sl=584.53 alert=retest1 |

### Cycle 78 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 554.50 | 574.59 | 574.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 553.10 | 562.14 | 568.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 14:15:00 | 553.95 | 551.45 | 558.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 15:00:00 | 553.95 | 551.45 | 558.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 549.10 | 550.91 | 557.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 545.85 | 549.94 | 555.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 545.65 | 547.03 | 551.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:00:00 | 545.75 | 545.50 | 548.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 546.10 | 545.71 | 548.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 547.45 | 546.06 | 548.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:45:00 | 550.40 | 546.06 | 548.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 553.90 | 547.63 | 548.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 553.90 | 547.63 | 548.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 555.20 | 549.14 | 549.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 555.25 | 549.14 | 549.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 555.30 | 549.14 | 549.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 555.30 | 549.14 | 549.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 552.80 | 549.87 | 549.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 552.80 | 549.87 | 549.52 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 546.30 | 549.71 | 550.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 545.50 | 548.87 | 549.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 546.75 | 546.19 | 547.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:00:00 | 546.75 | 546.19 | 547.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 547.90 | 546.53 | 547.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 547.45 | 546.53 | 547.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 550.80 | 547.39 | 548.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 550.80 | 547.39 | 548.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 547.65 | 547.44 | 548.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 546.85 | 547.44 | 548.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 545.90 | 546.20 | 547.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:15:00 | 546.50 | 546.29 | 547.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:30:00 | 546.35 | 545.72 | 546.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 536.10 | 534.82 | 539.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 536.10 | 534.82 | 539.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 544.05 | 536.78 | 539.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-12 14:15:00 | 542.30 | 540.62 | 540.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 542.30 | 540.62 | 540.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 546.65 | 543.04 | 542.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 571.10 | 572.07 | 565.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 571.10 | 572.07 | 565.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 567.10 | 571.07 | 565.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 567.10 | 571.07 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 567.70 | 570.40 | 566.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:15:00 | 564.95 | 570.40 | 566.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 564.45 | 569.21 | 565.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 564.45 | 569.21 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 563.50 | 568.07 | 565.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 563.00 | 568.07 | 565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 561.70 | 566.79 | 565.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 557.00 | 566.79 | 565.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 558.80 | 563.76 | 564.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 552.55 | 560.75 | 562.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 554.05 | 551.11 | 554.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 555.85 | 551.11 | 554.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 557.30 | 552.34 | 554.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 558.75 | 552.34 | 554.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 552.65 | 552.41 | 554.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 555.00 | 552.41 | 554.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 556.15 | 553.15 | 554.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 556.15 | 553.15 | 554.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 559.90 | 554.50 | 555.30 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 564.90 | 556.58 | 556.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 568.20 | 558.91 | 557.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 558.00 | 558.72 | 557.33 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 554.85 | 557.08 | 557.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 551.60 | 555.98 | 556.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 551.60 | 550.24 | 552.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 552.15 | 550.24 | 552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 552.00 | 549.90 | 551.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 552.00 | 549.90 | 551.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 550.05 | 549.93 | 551.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 547.15 | 549.93 | 551.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 549.50 | 549.95 | 551.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 553.85 | 550.73 | 551.49 | SL hit (close>static) qty=1.00 sl=552.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 559.80 | 553.31 | 552.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 561.60 | 557.70 | 555.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 556.00 | 558.02 | 556.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 554.05 | 558.02 | 556.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 553.95 | 557.21 | 555.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 553.95 | 557.21 | 555.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 552.70 | 556.30 | 555.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 553.60 | 556.30 | 555.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 550.95 | 554.55 | 554.90 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 569.60 | 557.56 | 556.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 579.70 | 568.21 | 564.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 576.75 | 577.19 | 572.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:15:00 | 575.95 | 577.19 | 572.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 571.10 | 577.04 | 574.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 573.25 | 577.04 | 574.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 571.70 | 575.97 | 574.12 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 571.50 | 572.90 | 573.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 567.55 | 571.69 | 572.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 571.75 | 571.70 | 572.40 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 576.00 | 572.51 | 572.35 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 571.05 | 572.15 | 572.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 568.85 | 571.27 | 571.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 569.25 | 566.94 | 568.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 569.25 | 566.94 | 568.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 565.10 | 566.57 | 568.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 558.90 | 566.57 | 568.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 564.05 | 565.63 | 567.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 13:30:00 | 563.95 | 565.89 | 567.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 570.10 | 566.73 | 567.52 | SL hit (close>static) qty=1.00 sl=570.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 572.65 | 568.30 | 568.12 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 09:15:00 | 562.00 | 567.96 | 568.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 560.30 | 563.04 | 564.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 544.95 | 541.05 | 547.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 541.65 | 541.05 | 547.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 547.55 | 542.35 | 547.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 547.55 | 542.35 | 547.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 548.60 | 543.60 | 547.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 548.60 | 543.60 | 547.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 550.90 | 545.06 | 547.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 550.90 | 545.06 | 547.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 553.45 | 549.02 | 548.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 563.50 | 554.91 | 552.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 15:15:00 | 590.00 | 590.41 | 585.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:15:00 | 586.40 | 590.41 | 585.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 585.30 | 589.39 | 585.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 585.00 | 589.39 | 585.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 589.65 | 589.44 | 585.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 585.00 | 589.44 | 585.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 595.80 | 598.27 | 595.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 595.05 | 598.27 | 595.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 595.85 | 597.79 | 595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 595.50 | 597.79 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 596.90 | 597.36 | 595.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 596.80 | 597.36 | 595.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 595.50 | 596.98 | 595.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 593.45 | 596.98 | 595.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 590.40 | 595.67 | 595.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 590.40 | 595.67 | 595.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 592.40 | 595.01 | 594.84 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 589.15 | 593.84 | 594.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 588.10 | 592.69 | 593.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 585.15 | 584.72 | 587.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 584.25 | 584.70 | 587.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 584.00 | 584.56 | 586.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 580.65 | 584.14 | 585.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 570.25 | 567.20 | 566.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 570.25 | 567.20 | 566.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 14:15:00 | 579.60 | 570.69 | 568.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 574.60 | 576.11 | 573.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 570.65 | 576.11 | 573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 573.35 | 575.56 | 573.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 570.30 | 575.56 | 573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 571.00 | 574.65 | 573.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 570.20 | 574.65 | 573.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 569.95 | 573.71 | 572.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 569.80 | 573.71 | 572.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 568.35 | 571.92 | 572.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 566.50 | 570.84 | 571.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 570.40 | 569.74 | 570.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 570.40 | 569.74 | 570.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 570.65 | 569.92 | 570.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 571.00 | 569.92 | 570.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 569.55 | 569.85 | 570.70 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 572.90 | 570.90 | 570.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 576.30 | 571.98 | 571.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 571.30 | 572.60 | 571.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 570.65 | 572.60 | 571.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 574.45 | 572.97 | 571.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 575.35 | 573.09 | 572.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 575.15 | 573.72 | 572.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 576.65 | 574.30 | 572.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 568.55 | 573.03 | 572.87 | SL hit (close<static) qty=1.00 sl=570.40 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 567.80 | 571.99 | 572.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 566.05 | 570.80 | 571.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 568.85 | 567.07 | 569.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 569.45 | 567.07 | 569.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 565.20 | 562.53 | 565.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 565.20 | 562.53 | 565.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 562.40 | 562.51 | 565.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 563.75 | 562.51 | 565.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 569.60 | 563.92 | 565.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 569.60 | 563.92 | 565.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 571.80 | 565.50 | 566.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 571.30 | 565.50 | 566.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 572.00 | 566.80 | 566.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 578.05 | 569.05 | 567.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 572.40 | 574.10 | 571.66 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 564.70 | 570.20 | 570.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 562.00 | 568.56 | 569.75 | Break + close below crossover candle low |

### Cycle 101 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 578.80 | 570.61 | 570.58 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 567.50 | 570.45 | 570.65 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 571.90 | 570.66 | 570.64 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 570.35 | 570.60 | 570.61 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 571.50 | 570.78 | 570.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 572.60 | 571.22 | 570.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 570.95 | 571.17 | 570.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 561.00 | 571.17 | 570.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 564.00 | 569.74 | 570.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 552.60 | 564.10 | 567.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 551.20 | 550.41 | 557.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 541.90 | 545.10 | 548.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 542.40 | 543.97 | 547.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 545.45 | 538.22 | 537.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 545.45 | 538.22 | 537.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 551.40 | 541.62 | 539.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 546.30 | 546.44 | 543.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 549.50 | 547.12 | 544.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 544.00 | 546.78 | 545.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 544.00 | 546.78 | 545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 543.15 | 546.05 | 544.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 543.15 | 546.05 | 544.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 538.30 | 543.65 | 543.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 536.25 | 542.17 | 543.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 530.20 | 529.34 | 532.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 530.20 | 529.34 | 532.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 531.70 | 529.77 | 532.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 532.40 | 529.77 | 532.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 532.70 | 530.36 | 532.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 532.70 | 530.36 | 532.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 535.45 | 531.38 | 532.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 535.45 | 531.38 | 532.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 534.55 | 532.01 | 532.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 532.30 | 532.01 | 532.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 528.90 | 531.39 | 532.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 527.95 | 531.39 | 532.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 526.10 | 530.00 | 531.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 527.45 | 529.64 | 531.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:15:00 | 526.90 | 529.64 | 531.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 534.75 | 528.63 | 529.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 534.75 | 528.63 | 529.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 531.30 | 529.17 | 529.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:00:00 | 527.50 | 529.70 | 530.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 534.00 | 530.58 | 530.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 534.00 | 530.58 | 530.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 15:15:00 | 535.70 | 531.77 | 530.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 541.90 | 541.95 | 538.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 11:30:00 | 541.45 | 541.95 | 538.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 545.70 | 542.71 | 539.94 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 534.50 | 540.51 | 541.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 532.70 | 538.95 | 540.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 542.45 | 539.41 | 540.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 542.45 | 539.41 | 540.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 538.70 | 539.27 | 540.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 541.55 | 539.27 | 540.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 540.90 | 539.60 | 540.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 534.80 | 539.60 | 540.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 536.85 | 538.96 | 539.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:00:00 | 536.60 | 538.73 | 539.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 534.00 | 539.27 | 539.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 535.85 | 533.58 | 534.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 535.55 | 533.58 | 534.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 535.45 | 533.96 | 534.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 536.85 | 533.96 | 534.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 538.35 | 534.83 | 535.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 538.35 | 534.83 | 535.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 538.90 | 535.65 | 535.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 538.90 | 535.65 | 535.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 540.20 | 536.56 | 535.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 540.80 | 545.78 | 542.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 540.80 | 545.78 | 542.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 539.20 | 544.46 | 542.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 543.35 | 544.46 | 542.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 543.65 | 542.55 | 542.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 536.80 | 543.49 | 543.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 536.80 | 543.49 | 543.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 535.65 | 540.31 | 541.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 537.00 | 536.40 | 538.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 537.00 | 536.40 | 538.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 535.10 | 536.08 | 538.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 533.80 | 535.82 | 537.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 533.00 | 534.74 | 536.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:15:00 | 532.10 | 535.01 | 535.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:30:00 | 533.15 | 533.66 | 533.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 534.40 | 533.81 | 533.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 534.25 | 533.81 | 533.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 536.40 | 534.32 | 534.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 536.40 | 534.32 | 534.19 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 530.60 | 533.97 | 534.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 530.10 | 532.52 | 533.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 516.75 | 516.51 | 521.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 516.85 | 516.51 | 521.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 518.55 | 516.96 | 520.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 519.10 | 516.96 | 520.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 518.70 | 517.31 | 520.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 516.60 | 517.31 | 520.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 529.00 | 518.91 | 518.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 529.00 | 518.91 | 518.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 539.15 | 528.06 | 523.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 533.50 | 533.74 | 528.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 533.50 | 533.74 | 528.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 532.45 | 532.52 | 530.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 535.90 | 532.62 | 530.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 12:15:00 | 529.50 | 532.00 | 531.55 | SL hit (close<static) qty=1.00 sl=529.55 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 525.50 | 530.18 | 530.77 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 533.05 | 530.88 | 530.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 533.80 | 531.73 | 531.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 532.70 | 533.26 | 532.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 536.45 | 534.56 | 533.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 532.00 | 533.25 | 533.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 532.00 | 533.25 | 533.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 527.70 | 532.14 | 532.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 532.35 | 530.55 | 531.76 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 533.00 | 532.33 | 532.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 536.70 | 533.20 | 532.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 539.80 | 541.02 | 538.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 539.80 | 541.02 | 538.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 539.80 | 540.78 | 538.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 540.90 | 540.52 | 538.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 540.10 | 540.44 | 539.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 537.25 | 539.80 | 538.88 | SL hit (close<static) qty=1.00 sl=537.50 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 540.55 | 543.33 | 543.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 539.35 | 542.30 | 542.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 539.75 | 539.20 | 540.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 539.60 | 539.20 | 540.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 550.80 | 540.93 | 540.64 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 532.25 | 540.08 | 540.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 530.85 | 535.49 | 538.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 536.75 | 533.94 | 536.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 536.75 | 533.94 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 535.00 | 534.15 | 536.36 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 549.85 | 537.85 | 537.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 566.50 | 553.16 | 547.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 575.20 | 577.84 | 572.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 575.20 | 577.84 | 572.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 584.90 | 587.68 | 584.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 583.65 | 587.68 | 584.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 581.30 | 586.40 | 583.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 581.30 | 586.40 | 583.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 580.50 | 585.22 | 583.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 580.50 | 585.22 | 583.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 585.35 | 585.25 | 583.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 587.90 | 585.80 | 584.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 578.40 | 584.78 | 585.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 578.40 | 584.78 | 585.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 576.50 | 579.97 | 582.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 12:15:00 | 571.50 | 570.26 | 574.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:00:00 | 571.50 | 570.26 | 574.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 571.20 | 570.45 | 574.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 569.70 | 572.45 | 574.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 568.10 | 571.90 | 573.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 541.22 | 548.38 | 552.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 539.70 | 548.38 | 552.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 540.00 | 538.31 | 543.81 | SL hit (close>ema200) qty=0.50 sl=538.31 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 547.65 | 544.89 | 544.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 549.10 | 545.73 | 545.16 | Break + close above crossover candle high |

### Cycle 126 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 538.00 | 544.76 | 544.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 532.95 | 537.13 | 538.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 535.80 | 534.85 | 536.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 535.80 | 534.85 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 536.00 | 535.08 | 536.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 535.75 | 535.08 | 536.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 538.30 | 535.72 | 536.61 | SL hit (close>static) qty=1.00 sl=537.65 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 537.95 | 537.09 | 537.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 542.05 | 538.44 | 537.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 543.30 | 543.48 | 541.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 543.30 | 543.48 | 541.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 544.90 | 544.70 | 542.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 549.80 | 546.02 | 543.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 540.95 | 547.00 | 547.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 540.95 | 547.00 | 547.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 542.59 | 544.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 533.20 | 530.67 | 535.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 533.20 | 530.67 | 535.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 532.15 | 531.08 | 534.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 534.35 | 531.08 | 534.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 536.25 | 532.11 | 534.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 536.25 | 532.11 | 534.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 535.00 | 532.69 | 534.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 533.85 | 532.69 | 534.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 507.16 | 514.78 | 517.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 513.20 | 512.11 | 514.51 | SL hit (close>ema200) qty=0.50 sl=512.11 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 517.95 | 515.96 | 515.71 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 514.90 | 518.28 | 518.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 13:15:00 | 509.45 | 513.91 | 515.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 465.00 | 464.50 | 470.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:30:00 | 465.00 | 464.50 | 470.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 455.45 | 461.48 | 466.18 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 471.30 | 466.99 | 466.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 478.70 | 470.89 | 468.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 474.45 | 477.23 | 474.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 472.50 | 477.23 | 474.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 473.95 | 476.57 | 474.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 473.95 | 476.57 | 474.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 474.75 | 476.21 | 474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 474.35 | 476.21 | 474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 473.10 | 475.59 | 474.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 474.00 | 475.59 | 474.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 473.55 | 475.18 | 474.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:15:00 | 472.50 | 475.18 | 474.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 472.95 | 474.73 | 474.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 472.45 | 474.73 | 474.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 473.85 | 474.56 | 474.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 462.10 | 474.56 | 474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 464.25 | 472.50 | 473.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 458.10 | 461.92 | 465.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 457.45 | 456.78 | 460.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 457.45 | 456.78 | 460.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 460.80 | 457.58 | 460.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 463.90 | 457.58 | 460.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 463.60 | 458.79 | 460.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 464.35 | 458.79 | 460.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 463.45 | 459.72 | 461.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 463.50 | 459.72 | 461.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 463.30 | 461.08 | 461.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 462.55 | 461.08 | 461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 463.90 | 461.64 | 461.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 460.40 | 461.52 | 461.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 460.05 | 461.52 | 461.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:45:00 | 459.60 | 459.58 | 460.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 459.75 | 459.76 | 460.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 464.25 | 460.66 | 460.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 464.25 | 460.66 | 460.61 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 459.40 | 461.52 | 461.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 457.65 | 459.61 | 460.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 460.80 | 457.68 | 458.50 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 461.70 | 459.17 | 459.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 464.30 | 461.29 | 460.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 461.20 | 461.94 | 460.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 462.55 | 461.94 | 460.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 460.70 | 461.73 | 460.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 460.70 | 461.73 | 460.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 461.60 | 461.70 | 460.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 462.30 | 461.82 | 461.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:45:00 | 463.65 | 462.45 | 461.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 462.85 | 462.53 | 461.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 457.30 | 460.97 | 461.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 457.30 | 460.97 | 461.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 14:15:00 | 455.40 | 459.26 | 460.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 461.35 | 459.48 | 460.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 461.35 | 459.48 | 460.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 461.10 | 459.80 | 460.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 459.15 | 459.35 | 459.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 436.19 | 454.90 | 457.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 451.00 | 450.07 | 453.16 | SL hit (close>ema200) qty=0.50 sl=450.07 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 437.00 | 435.96 | 435.96 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 433.70 | 435.80 | 435.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 429.90 | 433.92 | 434.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 405.00 | 400.96 | 409.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 405.00 | 400.96 | 409.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 416.85 | 404.61 | 409.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 414.40 | 404.61 | 409.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 416.55 | 407.00 | 410.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 415.30 | 407.00 | 410.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 417.35 | 411.78 | 411.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 417.35 | 411.78 | 411.74 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 410.00 | 413.21 | 413.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 409.05 | 412.05 | 412.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 415.80 | 412.63 | 412.99 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 415.70 | 413.24 | 413.23 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 405.85 | 413.76 | 413.86 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 415.20 | 411.37 | 411.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 423.20 | 414.51 | 412.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 412.70 | 420.48 | 417.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 413.70 | 420.48 | 417.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 414.50 | 419.28 | 417.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 413.25 | 419.28 | 417.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 416.50 | 417.79 | 417.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 416.50 | 417.79 | 417.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 414.30 | 417.09 | 416.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 414.30 | 417.09 | 416.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 414.20 | 416.51 | 416.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 408.25 | 414.86 | 415.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 409.90 | 409.71 | 412.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 409.90 | 409.71 | 412.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 414.00 | 410.57 | 412.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 417.80 | 410.57 | 412.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 417.40 | 411.93 | 413.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 413.60 | 412.93 | 413.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 416.80 | 413.70 | 413.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 416.80 | 413.70 | 413.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 417.70 | 415.09 | 414.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 412.65 | 415.05 | 414.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 10:30:00 | 415.65 | 415.93 | 414.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 15:15:00 | 457.22 | 449.91 | 443.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 471.95 | 475.82 | 476.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 470.70 | 474.80 | 475.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 463.70 | 463.15 | 466.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:30:00 | 464.40 | 463.15 | 466.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 463.15 | 463.55 | 465.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 465.40 | 463.55 | 465.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 466.75 | 464.19 | 465.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 462.95 | 464.19 | 465.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 460.50 | 463.45 | 465.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 458.85 | 460.17 | 462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 457.40 | 459.74 | 462.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 458.90 | 459.72 | 461.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 468.95 | 462.95 | 462.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 468.95 | 462.95 | 462.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 471.55 | 465.72 | 463.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 468.90 | 470.61 | 467.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 12:00:00 | 468.90 | 470.61 | 467.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 467.10 | 469.99 | 468.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 474.90 | 468.67 | 468.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 471.75 | 470.94 | 469.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 472.35 | 471.15 | 469.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 12:15:00 | 518.93 | 502.94 | 491.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 14:15:00 | 488.35 | 2024-05-27 13:15:00 | 490.75 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-05-24 11:45:00 | 488.20 | 2024-05-27 13:15:00 | 490.75 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-05-27 12:15:00 | 488.50 | 2024-05-27 13:15:00 | 490.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-06-27 13:45:00 | 497.70 | 2024-06-27 14:15:00 | 501.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-04 11:15:00 | 515.45 | 2024-07-08 13:15:00 | 509.60 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-04 13:15:00 | 514.00 | 2024-07-08 13:15:00 | 509.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-07-04 14:00:00 | 514.40 | 2024-07-08 13:15:00 | 509.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-07-05 09:15:00 | 514.90 | 2024-07-08 13:15:00 | 509.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-07-16 09:15:00 | 521.35 | 2024-07-16 10:15:00 | 520.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-07-26 09:15:00 | 535.00 | 2024-08-05 09:15:00 | 546.65 | STOP_HIT | 1.00 | 2.18% |
| SELL | retest2 | 2024-08-06 14:30:00 | 540.65 | 2024-08-13 10:15:00 | 535.75 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-08-06 15:00:00 | 539.90 | 2024-08-13 10:15:00 | 535.75 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2024-08-21 10:15:00 | 580.40 | 2024-08-23 14:15:00 | 565.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-08-22 09:15:00 | 595.50 | 2024-08-23 14:15:00 | 565.60 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest2 | 2024-09-04 12:15:00 | 593.20 | 2024-09-18 09:15:00 | 615.00 | STOP_HIT | 1.00 | 3.67% |
| BUY | retest2 | 2024-09-06 11:15:00 | 592.15 | 2024-09-18 09:15:00 | 615.00 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2024-09-26 09:15:00 | 611.90 | 2024-09-27 09:15:00 | 619.15 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-10-09 11:15:00 | 574.80 | 2024-10-15 10:15:00 | 584.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-10-10 11:00:00 | 574.90 | 2024-10-15 10:15:00 | 584.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-10-10 11:45:00 | 574.85 | 2024-10-15 10:15:00 | 584.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-10 14:30:00 | 574.65 | 2024-10-15 10:15:00 | 584.70 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-10-21 11:30:00 | 555.70 | 2024-10-28 12:15:00 | 545.20 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2024-10-22 09:15:00 | 553.50 | 2024-10-28 12:15:00 | 545.20 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2024-11-08 11:30:00 | 507.80 | 2024-11-18 14:15:00 | 482.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:30:00 | 507.80 | 2024-11-22 09:15:00 | 472.00 | STOP_HIT | 0.50 | 7.05% |
| SELL | retest2 | 2024-12-12 09:30:00 | 472.75 | 2024-12-16 11:15:00 | 475.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-26 09:15:00 | 441.90 | 2024-12-30 10:15:00 | 444.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-26 09:45:00 | 442.00 | 2024-12-30 12:15:00 | 446.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-26 12:00:00 | 441.60 | 2024-12-30 12:15:00 | 446.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-12-27 11:45:00 | 442.00 | 2024-12-30 12:15:00 | 446.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-30 09:15:00 | 438.75 | 2024-12-30 12:15:00 | 446.40 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-12-31 14:45:00 | 448.70 | 2025-01-06 10:15:00 | 446.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-01-01 09:15:00 | 453.55 | 2025-01-06 10:15:00 | 446.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-08 09:15:00 | 449.30 | 2025-01-08 09:15:00 | 455.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-22 14:15:00 | 474.45 | 2025-01-28 10:15:00 | 472.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-02-20 10:00:00 | 490.45 | 2025-02-21 09:15:00 | 484.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-02-20 11:45:00 | 490.15 | 2025-02-21 09:15:00 | 484.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-20 14:00:00 | 489.90 | 2025-02-21 09:15:00 | 484.90 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-02-20 14:30:00 | 491.85 | 2025-02-21 09:15:00 | 484.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-02-21 11:30:00 | 488.30 | 2025-02-28 10:15:00 | 490.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-02-21 12:15:00 | 488.00 | 2025-02-28 10:15:00 | 490.30 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-02-21 13:45:00 | 488.30 | 2025-02-28 10:15:00 | 490.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-03-25 13:45:00 | 506.65 | 2025-03-25 14:15:00 | 501.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest1 | 2025-04-21 09:15:00 | 553.70 | 2025-04-22 14:15:00 | 581.39 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 09:15:00 | 553.70 | 2025-04-24 11:15:00 | 584.10 | STOP_HIT | 0.50 | 5.49% |
| SELL | retest2 | 2025-04-29 12:15:00 | 545.85 | 2025-05-05 10:15:00 | 552.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-04-30 11:15:00 | 545.65 | 2025-05-05 10:15:00 | 552.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-02 10:00:00 | 545.75 | 2025-05-05 10:15:00 | 552.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-05-02 10:45:00 | 546.10 | 2025-05-05 10:15:00 | 552.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-05-07 13:15:00 | 546.85 | 2025-05-12 14:15:00 | 542.30 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-05-07 14:30:00 | 545.90 | 2025-05-12 14:15:00 | 542.30 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-05-08 11:15:00 | 546.50 | 2025-05-12 14:15:00 | 542.30 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-05-08 12:30:00 | 546.35 | 2025-05-12 14:15:00 | 542.30 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-05-28 09:15:00 | 547.15 | 2025-05-28 10:15:00 | 553.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-28 10:15:00 | 549.50 | 2025-05-28 10:15:00 | 553.85 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-13 09:15:00 | 558.90 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 564.05 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-13 13:30:00 | 563.95 | 2025-06-13 14:15:00 | 570.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-09 12:15:00 | 584.25 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-07-09 13:00:00 | 584.00 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-07-10 09:45:00 | 580.65 | 2025-07-17 11:15:00 | 570.25 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2025-07-24 12:15:00 | 575.35 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-24 14:00:00 | 575.15 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-24 15:00:00 | 576.65 | 2025-07-25 11:15:00 | 568.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-12 11:00:00 | 541.90 | 2025-08-18 12:15:00 | 545.45 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-12 12:30:00 | 542.40 | 2025-08-18 12:15:00 | 545.45 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-26 10:15:00 | 527.95 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-26 11:30:00 | 526.10 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-26 12:30:00 | 527.45 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-26 13:15:00 | 526.90 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-28 15:00:00 | 527.50 | 2025-08-29 13:15:00 | 534.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-08 09:15:00 | 534.80 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-08 09:45:00 | 536.85 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-08 14:00:00 | 536.60 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-09 09:15:00 | 534.00 | 2025-09-11 10:15:00 | 538.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-15 09:15:00 | 543.35 | 2025-09-16 14:15:00 | 536.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-15 14:00:00 | 543.65 | 2025-09-16 14:15:00 | 536.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-19 11:15:00 | 533.80 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-19 14:00:00 | 533.00 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-22 13:15:00 | 532.10 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-24 11:30:00 | 533.15 | 2025-09-24 13:15:00 | 536.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-30 11:15:00 | 516.60 | 2025-10-01 14:15:00 | 529.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-10-07 14:15:00 | 535.90 | 2025-10-08 12:15:00 | 529.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-14 12:00:00 | 536.45 | 2025-10-15 12:15:00 | 532.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-21 13:45:00 | 540.90 | 2025-10-23 09:15:00 | 537.25 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-21 14:30:00 | 540.10 | 2025-10-23 09:15:00 | 537.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-23 10:45:00 | 541.80 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-23 12:15:00 | 540.60 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-10-23 13:45:00 | 543.00 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-10-24 12:30:00 | 542.35 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-24 14:00:00 | 542.95 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-10-27 09:30:00 | 542.50 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-29 11:45:00 | 544.75 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-29 14:45:00 | 545.25 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-30 12:30:00 | 544.90 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-30 15:00:00 | 545.00 | 2025-10-31 14:15:00 | 540.55 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-20 12:30:00 | 587.90 | 2025-11-21 14:15:00 | 578.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-27 13:15:00 | 569.70 | 2025-12-08 11:15:00 | 541.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 15:15:00 | 568.10 | 2025-12-08 11:15:00 | 539.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 569.70 | 2025-12-09 12:15:00 | 540.00 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-27 15:15:00 | 568.10 | 2025-12-09 12:15:00 | 540.00 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-12-19 09:15:00 | 535.75 | 2025-12-19 09:15:00 | 538.30 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-24 12:00:00 | 549.80 | 2025-12-29 09:15:00 | 540.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-01 09:15:00 | 533.85 | 2026-01-12 09:15:00 | 507.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 533.85 | 2026-01-13 10:15:00 | 513.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2026-02-12 14:45:00 | 460.40 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-12 15:15:00 | 460.05 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-13 14:45:00 | 459.60 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-16 09:15:00 | 459.75 | 2026-02-16 09:15:00 | 464.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-24 14:00:00 | 462.30 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-25 09:45:00 | 463.65 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-25 11:00:00 | 462.85 | 2026-02-25 12:15:00 | 457.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-26 11:30:00 | 459.15 | 2026-03-02 09:15:00 | 436.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 459.15 | 2026-03-02 15:15:00 | 451.00 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2026-03-17 11:15:00 | 415.30 | 2026-03-17 13:15:00 | 417.35 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-04-01 11:45:00 | 413.60 | 2026-04-01 12:15:00 | 416.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-04-02 10:30:00 | 415.65 | 2026-04-10 15:15:00 | 457.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-29 10:00:00 | 458.85 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-29 10:30:00 | 457.40 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-30 09:30:00 | 458.90 | 2026-04-30 11:15:00 | 468.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-05-05 15:15:00 | 474.90 | 2026-05-08 12:15:00 | 518.93 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2026-05-06 11:30:00 | 471.75 | 2026-05-08 12:15:00 | 519.59 | TARGET_HIT | 1.00 | 10.14% |
