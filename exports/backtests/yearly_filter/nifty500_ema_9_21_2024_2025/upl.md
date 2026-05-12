# UPL Ltd. (UPL)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 644.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 105 |
| ALERT2 | 105 |
| ALERT2_SKIP | 50 |
| ALERT3 | 277 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 115 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 120 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 97
- **Target hits / Stop hits / Partials:** 0 / 120 / 5
- **Avg / median % per leg:** -0.28% / -0.87%
- **Sum % (uncompounded):** -34.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 17 | 23.6% | 0 | 71 | 1 | -0.22% | -16.0% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 0 | 5 | 1 | 1.91% | 11.5% |
| BUY @ 3rd Alert (retest2) | 66 | 12 | 18.2% | 0 | 66 | 0 | -0.42% | -27.4% |
| SELL (all) | 53 | 11 | 20.8% | 0 | 49 | 4 | -0.35% | -18.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 11 | 20.8% | 0 | 49 | 4 | -0.35% | -18.4% |
| retest1 (combined) | 6 | 5 | 83.3% | 0 | 5 | 1 | 1.91% | 11.5% |
| retest2 (combined) | 119 | 23 | 19.3% | 0 | 115 | 4 | -0.39% | -45.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 488.40 | 491.93 | 492.08 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 498.76 | 492.53 | 492.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 505.95 | 497.51 | 495.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 502.12 | 502.89 | 499.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:00:00 | 502.12 | 502.89 | 499.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 499.96 | 502.13 | 499.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 499.96 | 502.13 | 499.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 498.71 | 501.45 | 499.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 496.94 | 501.45 | 499.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 496.26 | 500.41 | 499.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 496.26 | 500.41 | 499.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 496.17 | 498.88 | 498.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 492.14 | 496.57 | 497.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 490.80 | 489.65 | 492.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 490.80 | 489.65 | 492.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 504.08 | 492.31 | 492.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 504.08 | 492.31 | 492.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 504.23 | 494.69 | 493.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 508.40 | 497.44 | 495.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 487.20 | 501.13 | 498.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 474.73 | 495.85 | 496.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 473.43 | 491.37 | 494.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 490.03 | 485.42 | 489.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 500.92 | 488.52 | 490.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 500.92 | 488.52 | 490.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 502.84 | 491.38 | 491.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 502.84 | 491.38 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 502.55 | 493.62 | 492.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 503.75 | 495.64 | 493.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 512.86 | 515.58 | 511.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 513.39 | 515.14 | 511.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:30:00 | 511.13 | 515.14 | 511.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 533.82 | 529.46 | 526.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 537.18 | 529.46 | 526.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 538.28 | 532.54 | 531.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 535.21 | 533.88 | 532.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 534.78 | 534.32 | 532.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 539.72 | 535.38 | 533.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 544.56 | 537.23 | 534.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:00:00 | 543.79 | 544.03 | 539.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:00:00 | 543.60 | 543.94 | 540.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 543.41 | 543.67 | 541.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 535.59 | 541.81 | 540.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 530.51 | 541.81 | 540.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 547.25 | 547.26 | 544.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 546.34 | 547.26 | 544.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 544.32 | 546.67 | 545.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:30:00 | 549.36 | 547.07 | 545.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 548.50 | 547.65 | 546.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:00:00 | 548.59 | 547.84 | 546.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 548.97 | 547.85 | 546.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 546.96 | 547.67 | 546.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 545.81 | 547.67 | 546.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 545.33 | 547.20 | 546.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 545.33 | 547.20 | 546.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 541.54 | 546.07 | 546.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 541.54 | 546.07 | 546.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 539.00 | 544.66 | 545.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 548.26 | 544.62 | 545.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 546.29 | 544.95 | 545.49 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 547.54 | 545.70 | 545.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 548.69 | 546.68 | 546.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 13:15:00 | 547.49 | 548.00 | 547.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 14:00:00 | 547.49 | 548.00 | 547.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 545.33 | 548.97 | 548.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 545.33 | 548.97 | 548.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 540.77 | 547.33 | 547.41 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 551.56 | 547.67 | 547.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 555.40 | 549.79 | 548.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 549.60 | 550.03 | 548.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 547.06 | 549.44 | 548.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 547.06 | 549.44 | 548.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 547.25 | 549.00 | 548.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 546.58 | 549.00 | 548.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 548.16 | 548.75 | 548.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:45:00 | 549.41 | 548.83 | 548.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:30:00 | 549.89 | 548.83 | 548.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 549.45 | 548.81 | 548.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 546.91 | 548.53 | 548.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 546.91 | 548.53 | 548.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 544.03 | 547.63 | 548.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 545.62 | 544.83 | 546.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:15:00 | 545.09 | 544.83 | 546.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 544.32 | 544.73 | 546.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 547.25 | 544.73 | 546.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 546.91 | 545.16 | 546.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 546.91 | 545.16 | 546.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 545.09 | 545.15 | 546.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 546.82 | 545.15 | 546.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 532.19 | 540.89 | 543.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 529.50 | 540.89 | 543.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 530.41 | 537.73 | 541.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 541.88 | 539.31 | 539.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 541.88 | 539.31 | 539.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 543.27 | 541.39 | 540.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 540.63 | 541.24 | 540.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 539.67 | 540.93 | 540.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 539.67 | 540.93 | 540.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 538.14 | 540.37 | 540.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 538.14 | 540.37 | 540.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 535.55 | 539.41 | 539.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 528.73 | 534.91 | 536.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 526.05 | 525.78 | 530.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 526.05 | 525.78 | 530.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 531.85 | 527.00 | 530.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 530.32 | 527.00 | 530.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 529.45 | 527.49 | 530.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 531.95 | 527.49 | 530.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 527.20 | 527.43 | 530.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 522.98 | 526.33 | 529.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 522.12 | 525.67 | 528.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 521.30 | 516.40 | 516.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 521.30 | 516.40 | 516.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 522.40 | 517.60 | 516.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 544.27 | 546.19 | 541.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 11:00:00 | 544.27 | 546.19 | 541.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 540.53 | 545.05 | 541.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 540.53 | 545.05 | 541.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 542.36 | 544.52 | 541.58 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 530.89 | 539.21 | 539.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 514.78 | 532.97 | 536.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 523.89 | 515.38 | 522.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 524.75 | 517.26 | 522.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 524.75 | 517.26 | 522.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 523.60 | 518.52 | 522.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 522.79 | 518.52 | 522.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 526.14 | 520.82 | 520.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 526.14 | 520.82 | 520.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 527.15 | 522.09 | 521.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 524.27 | 524.80 | 523.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 524.27 | 524.80 | 523.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 524.99 | 530.03 | 527.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:30:00 | 536.36 | 531.54 | 529.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 523.89 | 531.01 | 531.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 523.89 | 531.01 | 531.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 521.16 | 529.04 | 530.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 523.70 | 522.76 | 525.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 12:00:00 | 523.70 | 522.76 | 525.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 522.26 | 522.66 | 525.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 525.09 | 522.66 | 525.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 528.11 | 523.75 | 525.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 528.11 | 523.75 | 525.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 530.85 | 525.17 | 526.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 530.85 | 525.17 | 526.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 533.91 | 527.70 | 527.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 538.76 | 533.12 | 530.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 550.89 | 553.19 | 548.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 549.98 | 552.16 | 549.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 549.98 | 552.16 | 549.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 549.55 | 551.64 | 549.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 551.52 | 551.64 | 549.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:00:00 | 551.09 | 551.06 | 550.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:30:00 | 551.76 | 551.18 | 550.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 551.28 | 554.30 | 554.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 551.28 | 554.30 | 554.56 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 559.38 | 554.57 | 554.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 568.54 | 557.37 | 555.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 577.94 | 578.04 | 574.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:30:00 | 577.46 | 578.04 | 574.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 588.74 | 589.95 | 584.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 588.74 | 589.95 | 584.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 589.55 | 589.87 | 585.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 588.11 | 589.87 | 585.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 584.80 | 590.21 | 587.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 583.51 | 590.21 | 587.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 585.91 | 589.35 | 586.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 581.73 | 589.35 | 586.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 580.77 | 584.90 | 585.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 576.89 | 583.29 | 584.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 587.78 | 583.37 | 584.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 592.04 | 585.11 | 584.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 594.44 | 589.15 | 587.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 589.41 | 590.17 | 588.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:45:00 | 589.98 | 590.17 | 588.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 591.71 | 590.48 | 588.46 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 586.34 | 587.68 | 587.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 585.23 | 586.93 | 587.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 587.92 | 586.97 | 587.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 589.45 | 587.46 | 587.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 588.98 | 587.46 | 587.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 589.93 | 587.96 | 587.73 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 586.19 | 587.83 | 587.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 582.50 | 586.45 | 587.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:45:00 | 586.58 | 586.63 | 587.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 588.40 | 586.98 | 587.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 588.40 | 586.98 | 587.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 589.31 | 587.45 | 587.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 589.31 | 587.45 | 587.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 588.50 | 587.66 | 587.64 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 585.33 | 587.19 | 587.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 582.21 | 585.56 | 586.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 571.71 | 570.77 | 576.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 571.71 | 570.77 | 576.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 573.10 | 571.16 | 575.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:15:00 | 568.11 | 571.81 | 574.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 566.72 | 571.38 | 574.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 581.25 | 572.02 | 571.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 581.25 | 572.02 | 571.61 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 570.85 | 574.41 | 574.65 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 577.85 | 574.64 | 574.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 582.21 | 576.16 | 575.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 588.35 | 593.53 | 590.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 584.23 | 591.67 | 590.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 584.23 | 591.67 | 590.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 580.34 | 589.40 | 589.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 580.34 | 589.40 | 589.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 580.68 | 587.66 | 588.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 573.92 | 581.58 | 584.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 558.04 | 556.70 | 562.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 10:30:00 | 556.98 | 556.70 | 562.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 567.44 | 558.12 | 560.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 566.24 | 558.12 | 560.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 563.46 | 559.19 | 560.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 561.64 | 560.21 | 561.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 561.64 | 560.61 | 561.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 561.92 | 561.18 | 561.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.56 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.56 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.82 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 533.15 | 532.97 | 538.78 | SL hit (close>ema200) qty=0.50 sl=532.97 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 513.34 | 510.17 | 509.98 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 504.51 | 509.20 | 509.62 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 513.00 | 509.85 | 509.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 516.46 | 511.17 | 510.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 524.37 | 529.17 | 524.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 523.70 | 528.07 | 524.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 523.70 | 528.07 | 524.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 529.50 | 528.36 | 524.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 530.65 | 528.79 | 525.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 533.77 | 528.92 | 526.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 530.32 | 530.10 | 527.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 14:15:00 | 535.69 | 538.89 | 538.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 535.69 | 538.89 | 538.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 532.48 | 537.61 | 538.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 500.34 | 511.03 | 517.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:15:00 | 500.25 | 498.65 | 504.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 498.52 | 501.81 | 504.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 513.43 | 506.08 | 505.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 513.43 | 506.08 | 505.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 517.32 | 511.31 | 508.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 530.61 | 531.53 | 525.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 09:30:00 | 530.08 | 531.53 | 525.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 529.12 | 531.05 | 525.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 524.75 | 531.05 | 525.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 558.00 | 548.78 | 541.81 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 539.55 | 547.27 | 547.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 538.95 | 544.52 | 546.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 544.70 | 544.18 | 545.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 544.95 | 544.33 | 545.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 544.95 | 544.33 | 545.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 544.55 | 544.38 | 545.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 542.15 | 544.38 | 545.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 547.60 | 545.02 | 545.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 547.60 | 545.02 | 545.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 548.55 | 545.73 | 545.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 550.00 | 545.73 | 545.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 554.25 | 547.43 | 546.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 556.00 | 552.76 | 550.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 555.25 | 564.43 | 561.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 555.25 | 562.59 | 560.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 553.85 | 562.59 | 560.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 557.90 | 559.62 | 559.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 557.90 | 559.62 | 559.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 559.60 | 559.57 | 559.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 565.30 | 560.38 | 559.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 552.35 | 559.89 | 560.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 552.35 | 559.89 | 560.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 550.60 | 554.14 | 556.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 563.90 | 555.10 | 556.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 562.35 | 556.55 | 556.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:30:00 | 565.00 | 556.55 | 556.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 560.00 | 557.24 | 556.93 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 552.00 | 555.84 | 556.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 545.50 | 553.44 | 555.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 545.25 | 544.32 | 547.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 545.25 | 544.32 | 547.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 550.45 | 545.55 | 548.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 550.45 | 545.55 | 548.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 549.65 | 546.37 | 548.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 547.10 | 546.37 | 548.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 547.35 | 546.57 | 547.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 547.55 | 546.57 | 547.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 549.15 | 547.09 | 547.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 549.15 | 547.09 | 547.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 548.30 | 547.33 | 547.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 547.00 | 547.33 | 547.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 550.60 | 547.93 | 548.07 | SL hit (close>static) qty=1.00 sl=549.40 alert=retest2 |

### Cycle 42 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 505.20 | 502.00 | 501.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 507.65 | 503.13 | 502.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 12:15:00 | 524.85 | 525.00 | 518.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 12:45:00 | 525.55 | 525.00 | 518.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 519.25 | 523.85 | 518.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 519.25 | 523.85 | 518.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 520.70 | 523.22 | 518.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 516.65 | 523.22 | 518.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 539.65 | 526.02 | 520.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 542.05 | 526.02 | 520.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 09:30:00 | 540.30 | 539.97 | 532.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 15:00:00 | 540.30 | 537.74 | 533.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:30:00 | 540.65 | 543.38 | 540.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 545.70 | 546.88 | 544.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 10:15:00 | 546.95 | 546.88 | 544.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 537.05 | 544.01 | 543.15 | SL hit (close<static) qty=1.00 sl=538.30 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 535.35 | 541.23 | 541.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 533.35 | 539.66 | 541.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 540.65 | 538.71 | 540.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 538.95 | 538.76 | 540.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 537.40 | 538.76 | 540.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 542.25 | 539.15 | 540.06 | SL hit (close>static) qty=1.00 sl=541.50 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 546.60 | 541.60 | 541.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 550.25 | 543.33 | 541.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 15:15:00 | 545.30 | 545.94 | 544.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:15:00 | 550.60 | 545.94 | 544.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 545.80 | 547.35 | 545.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:00:00 | 545.80 | 547.35 | 545.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 548.50 | 547.58 | 545.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 550.00 | 547.11 | 545.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 547.65 | 549.93 | 548.45 | SL hit (close<ema400) qty=1.00 sl=548.45 alert=retest1 |

### Cycle 45 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 544.30 | 548.73 | 549.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 538.55 | 546.17 | 547.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 543.40 | 540.37 | 543.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 553.00 | 543.09 | 544.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 553.00 | 543.09 | 544.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 554.50 | 545.37 | 545.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 555.05 | 545.37 | 545.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 555.00 | 547.30 | 546.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 556.75 | 549.19 | 547.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 12:15:00 | 554.10 | 554.44 | 551.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 13:00:00 | 554.10 | 554.44 | 551.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 551.50 | 553.64 | 551.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 552.00 | 553.64 | 551.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 550.40 | 552.99 | 551.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 544.75 | 552.99 | 551.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 538.95 | 550.19 | 550.38 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 549.15 | 545.46 | 545.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 553.40 | 547.87 | 546.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 564.00 | 565.67 | 558.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:15:00 | 570.95 | 565.67 | 558.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 14:15:00 | 599.50 | 580.74 | 569.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 599.60 | 601.31 | 589.08 | SL hit (close<ema200) qty=0.50 sl=601.31 alert=retest1 |

### Cycle 49 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 633.70 | 637.97 | 638.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 621.95 | 633.25 | 635.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 612.95 | 612.34 | 620.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 612.95 | 612.34 | 620.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 619.90 | 615.09 | 619.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 619.90 | 615.09 | 619.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 621.90 | 616.45 | 620.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 616.40 | 616.45 | 620.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 622.10 | 617.58 | 620.34 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 636.20 | 623.73 | 622.81 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 617.20 | 623.42 | 623.63 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 13:15:00 | 626.20 | 621.71 | 621.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 633.40 | 624.05 | 622.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 12:15:00 | 626.60 | 627.44 | 625.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-18 12:45:00 | 628.20 | 627.44 | 625.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 642.50 | 631.43 | 627.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 10:15:00 | 643.70 | 631.43 | 627.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:45:00 | 645.00 | 645.56 | 642.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 648.40 | 644.95 | 643.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 644.45 | 644.50 | 643.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 645.40 | 644.68 | 643.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 644.90 | 644.68 | 643.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 645.85 | 644.91 | 643.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 643.00 | 644.91 | 643.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 645.20 | 644.97 | 643.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 651.60 | 645.18 | 644.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 641.95 | 646.26 | 646.04 | SL hit (close<static) qty=1.00 sl=643.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 639.45 | 644.89 | 645.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 636.60 | 643.24 | 644.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 642.55 | 642.06 | 643.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 640.60 | 641.77 | 643.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 625.00 | 641.77 | 643.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 629.25 | 622.84 | 622.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 629.25 | 622.84 | 622.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 632.50 | 626.20 | 624.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:30:00 | 626.25 | 627.25 | 625.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 624.70 | 626.74 | 625.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 624.05 | 626.74 | 625.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 624.45 | 626.28 | 625.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 625.00 | 626.28 | 625.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 629.25 | 631.63 | 629.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 629.25 | 631.63 | 629.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 631.50 | 631.60 | 629.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 630.25 | 631.60 | 629.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 628.75 | 631.03 | 629.36 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 619.30 | 627.31 | 627.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 619.15 | 625.67 | 627.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 612.70 | 609.27 | 613.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 608.45 | 609.11 | 612.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 606.70 | 609.11 | 612.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 606.50 | 607.63 | 611.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 616.05 | 608.97 | 610.43 | SL hit (close>static) qty=1.00 sl=614.30 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 620.90 | 613.25 | 612.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 625.85 | 615.77 | 613.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 656.85 | 657.31 | 649.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 656.85 | 657.31 | 649.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 656.50 | 659.16 | 655.57 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 652.55 | 654.04 | 654.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 650.50 | 653.17 | 653.75 | Break + close below crossover candle low |

### Cycle 58 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 659.40 | 654.42 | 654.27 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 652.20 | 654.42 | 654.57 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 659.50 | 655.44 | 655.02 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 630.25 | 650.40 | 652.77 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 654.10 | 643.22 | 643.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 655.05 | 647.29 | 645.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 637.90 | 646.40 | 645.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 643.65 | 645.85 | 645.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:45:00 | 646.10 | 645.96 | 645.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 648.05 | 647.95 | 646.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 640.45 | 645.11 | 645.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 640.45 | 645.11 | 645.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 637.80 | 643.65 | 644.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 617.00 | 616.83 | 625.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 617.00 | 616.83 | 625.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 630.75 | 616.18 | 617.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 628.95 | 616.18 | 617.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 639.60 | 620.87 | 619.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 643.05 | 625.30 | 622.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 651.85 | 656.52 | 650.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:30:00 | 664.15 | 657.80 | 651.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 12:30:00 | 660.75 | 658.31 | 652.70 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 661.20 | 657.30 | 653.58 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 663.30 | 658.50 | 654.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:30:00 | 668.65 | 663.44 | 658.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 671.20 | 671.31 | 666.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 666.40 | 670.33 | 666.51 | SL hit (close<ema400) qty=1.00 sl=666.51 alert=retest1 |

### Cycle 65 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 662.80 | 673.55 | 674.04 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 677.35 | 672.16 | 672.07 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 668.45 | 672.51 | 672.76 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 675.75 | 672.92 | 672.70 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 671.00 | 672.64 | 672.68 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 686.00 | 674.94 | 673.70 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 675.70 | 682.50 | 682.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 670.00 | 680.00 | 681.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 664.05 | 662.73 | 670.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 664.05 | 662.73 | 670.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 669.95 | 664.75 | 670.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 669.95 | 664.75 | 670.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 673.65 | 666.53 | 670.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 673.65 | 666.53 | 670.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 681.95 | 669.61 | 671.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 692.55 | 669.61 | 671.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 689.70 | 673.63 | 673.16 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 646.70 | 672.08 | 674.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 642.80 | 658.44 | 666.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 645.10 | 641.18 | 649.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 645.45 | 641.18 | 649.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 642.65 | 642.92 | 646.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 644.55 | 642.92 | 646.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 643.95 | 642.52 | 644.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 643.95 | 642.52 | 644.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 642.80 | 642.58 | 644.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 646.50 | 642.58 | 644.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 653.15 | 644.69 | 645.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 653.15 | 644.69 | 645.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 650.40 | 645.83 | 645.74 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 644.90 | 645.76 | 645.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 640.20 | 644.65 | 645.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 640.00 | 634.37 | 637.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 638.00 | 635.10 | 637.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 635.80 | 636.09 | 637.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 637.60 | 632.95 | 632.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 637.60 | 632.95 | 632.40 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 630.80 | 632.36 | 632.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 627.10 | 631.31 | 631.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 630.55 | 630.52 | 631.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 630.55 | 630.52 | 631.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 630.05 | 630.43 | 631.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 629.25 | 630.43 | 631.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 631.30 | 630.60 | 631.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 631.30 | 630.60 | 631.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 630.10 | 630.50 | 631.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 627.35 | 630.50 | 631.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 625.05 | 629.41 | 630.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 623.05 | 629.41 | 630.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 633.55 | 629.53 | 629.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 633.55 | 629.53 | 629.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 635.20 | 630.67 | 629.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 632.95 | 633.65 | 632.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 632.95 | 633.65 | 632.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 644.25 | 644.73 | 642.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 644.80 | 644.75 | 642.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 639.00 | 642.21 | 642.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 639.00 | 642.21 | 642.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 637.20 | 639.75 | 640.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 641.30 | 636.99 | 638.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 636.80 | 636.95 | 638.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 635.45 | 636.65 | 637.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:00:00 | 635.85 | 636.49 | 637.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 639.75 | 635.38 | 635.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 639.75 | 635.38 | 635.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 643.10 | 637.66 | 636.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 645.80 | 646.11 | 642.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 645.80 | 646.11 | 642.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 643.30 | 645.71 | 643.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 643.95 | 645.71 | 643.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 641.90 | 644.95 | 642.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 641.90 | 644.95 | 642.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 639.80 | 643.92 | 642.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 637.85 | 643.92 | 642.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 638.10 | 642.75 | 642.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 637.90 | 642.75 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 637.90 | 641.78 | 641.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 633.75 | 639.48 | 640.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 635.50 | 633.59 | 636.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 635.50 | 633.59 | 636.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 632.00 | 632.29 | 634.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 635.00 | 633.00 | 634.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 637.30 | 633.86 | 634.87 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 639.25 | 635.57 | 635.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 639.75 | 636.40 | 635.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 633.00 | 645.43 | 643.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 632.10 | 642.77 | 642.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 630.00 | 642.77 | 642.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 634.40 | 641.09 | 641.92 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 645.75 | 641.51 | 641.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 15:15:00 | 646.20 | 643.53 | 642.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 681.80 | 683.07 | 676.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 682.10 | 683.07 | 676.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 679.45 | 680.74 | 678.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 679.45 | 680.74 | 678.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 677.20 | 680.03 | 678.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 677.20 | 680.03 | 678.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 680.90 | 680.20 | 678.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:45:00 | 677.90 | 680.20 | 678.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 677.75 | 679.68 | 678.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 677.75 | 679.68 | 678.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 672.00 | 678.14 | 678.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 670.55 | 676.63 | 677.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 674.35 | 674.31 | 676.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 674.35 | 674.31 | 676.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 656.00 | 655.52 | 659.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 652.20 | 654.63 | 659.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 661.20 | 656.51 | 658.15 | SL hit (close>static) qty=1.00 sl=660.70 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 671.95 | 661.68 | 660.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 679.00 | 672.45 | 669.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 687.10 | 687.39 | 680.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:00:00 | 687.10 | 687.39 | 680.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 694.85 | 689.21 | 684.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 697.30 | 689.21 | 684.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 719.45 | 724.10 | 724.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 719.45 | 724.10 | 724.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 714.30 | 720.62 | 722.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 10:15:00 | 712.25 | 711.30 | 715.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:30:00 | 710.90 | 711.30 | 715.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 711.30 | 694.46 | 704.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 711.30 | 694.46 | 704.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 708.00 | 697.17 | 704.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:00:00 | 705.25 | 698.79 | 704.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 716.15 | 704.26 | 706.22 | SL hit (close>static) qty=1.00 sl=712.40 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 711.35 | 707.22 | 707.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 12:15:00 | 714.15 | 709.31 | 708.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 711.40 | 713.97 | 711.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 712.45 | 713.67 | 711.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 712.45 | 713.67 | 711.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 709.05 | 712.75 | 711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 709.05 | 712.75 | 711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 706.65 | 711.53 | 710.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 705.70 | 711.53 | 710.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 707.15 | 710.65 | 710.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 707.15 | 710.65 | 710.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 704.70 | 709.46 | 709.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 700.30 | 707.00 | 708.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 706.05 | 700.91 | 704.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 704.55 | 701.63 | 704.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 702.10 | 701.63 | 704.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 697.35 | 689.51 | 688.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 697.35 | 689.51 | 688.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 702.80 | 693.67 | 691.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 712.95 | 713.17 | 708.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 712.95 | 713.17 | 708.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 709.80 | 712.02 | 709.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 708.80 | 712.02 | 709.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 713.60 | 712.34 | 710.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 710.30 | 712.34 | 710.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 734.00 | 718.68 | 714.24 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 715.65 | 721.52 | 722.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 704.80 | 717.29 | 720.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 716.60 | 715.41 | 718.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 716.60 | 715.41 | 718.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 719.50 | 716.23 | 718.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 715.60 | 716.10 | 718.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 730.00 | 718.39 | 718.82 | SL hit (close>static) qty=1.00 sl=722.20 alert=retest2 |

### Cycle 92 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 726.95 | 720.11 | 719.56 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 711.60 | 719.65 | 720.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 706.40 | 713.55 | 715.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 705.00 | 701.60 | 701.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 715.40 | 704.36 | 702.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 702.85 | 704.76 | 703.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 701.00 | 704.01 | 703.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 701.00 | 704.01 | 703.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 699.25 | 703.06 | 702.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 699.25 | 703.06 | 702.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 704.85 | 703.73 | 703.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 699.55 | 703.73 | 703.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 706.10 | 704.20 | 703.40 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 694.75 | 701.99 | 702.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 693.30 | 700.25 | 701.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 698.00 | 697.43 | 699.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 698.00 | 697.43 | 699.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 701.20 | 698.18 | 699.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 701.20 | 698.18 | 699.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 701.15 | 698.78 | 699.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 700.90 | 698.78 | 699.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 704.50 | 700.25 | 700.13 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 698.30 | 699.86 | 699.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 695.45 | 698.98 | 699.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 698.55 | 695.69 | 697.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 689.55 | 694.46 | 696.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 688.95 | 694.46 | 696.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 687.35 | 693.45 | 695.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 700.50 | 694.30 | 695.11 | SL hit (close>static) qty=1.00 sl=699.40 alert=retest2 |

### Cycle 98 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 701.30 | 695.70 | 695.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 702.05 | 697.87 | 696.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 699.25 | 703.78 | 701.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 693.85 | 701.80 | 701.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 693.85 | 701.80 | 701.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 698.35 | 701.11 | 700.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 699.35 | 700.68 | 700.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 698.20 | 700.19 | 700.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 698.20 | 700.19 | 700.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 693.50 | 698.85 | 699.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 678.25 | 676.86 | 681.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 678.25 | 676.86 | 681.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 655.80 | 652.81 | 657.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 653.85 | 652.74 | 656.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 654.90 | 654.22 | 655.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 661.25 | 656.79 | 656.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 661.25 | 656.79 | 656.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 670.15 | 660.60 | 658.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 679.85 | 680.32 | 676.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:45:00 | 679.45 | 680.32 | 676.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 680.45 | 679.99 | 677.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 681.05 | 679.99 | 677.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 676.45 | 679.32 | 677.38 | SL hit (close<static) qty=1.00 sl=677.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 671.40 | 675.87 | 676.14 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 677.95 | 675.74 | 675.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 681.70 | 676.93 | 676.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 673.90 | 676.65 | 676.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 672.65 | 675.85 | 676.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 681.20 | 676.61 | 676.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 670.50 | 675.39 | 676.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 669.50 | 675.39 | 676.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 669.65 | 674.12 | 675.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:00:00 | 669.80 | 671.42 | 673.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 678.00 | 674.18 | 673.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 678.00 | 674.18 | 673.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 682.15 | 675.77 | 674.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 679.75 | 679.80 | 677.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 678.15 | 679.47 | 677.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 676.80 | 679.47 | 677.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 679.75 | 679.53 | 678.03 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 674.25 | 677.13 | 677.20 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 680.00 | 677.28 | 677.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 683.25 | 679.92 | 678.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 678.90 | 679.96 | 679.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 679.40 | 681.54 | 680.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 679.40 | 681.54 | 680.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 674.55 | 680.14 | 679.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 674.55 | 680.14 | 679.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 674.15 | 678.94 | 679.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 673.60 | 677.87 | 678.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 679.75 | 674.67 | 676.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 678.60 | 675.46 | 676.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 679.05 | 675.46 | 676.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 679.85 | 676.65 | 676.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 679.10 | 676.65 | 676.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 679.95 | 677.31 | 677.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 681.80 | 678.76 | 677.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 716.20 | 718.72 | 712.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:45:00 | 716.00 | 718.72 | 712.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 721.55 | 728.31 | 725.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 721.55 | 728.31 | 725.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 725.50 | 727.75 | 725.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 726.25 | 725.72 | 725.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:45:00 | 727.60 | 727.90 | 726.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 760.00 | 761.95 | 762.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 760.00 | 761.95 | 762.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 758.75 | 761.31 | 761.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 752.30 | 756.54 | 757.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 750.25 | 756.13 | 757.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 750.25 | 751.07 | 753.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 750.50 | 747.41 | 749.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 748.25 | 747.58 | 749.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 747.00 | 747.61 | 749.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 746.95 | 748.17 | 749.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 758.30 | 750.51 | 750.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 758.30 | 750.51 | 750.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 759.15 | 752.24 | 750.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 759.20 | 759.41 | 756.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 759.20 | 759.41 | 756.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 759.15 | 759.36 | 756.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 757.80 | 759.36 | 756.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 762.90 | 759.85 | 757.56 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 755.25 | 757.68 | 757.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 754.00 | 756.94 | 757.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 748.00 | 747.19 | 750.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:45:00 | 748.00 | 747.19 | 750.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 747.50 | 745.20 | 748.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 754.80 | 745.20 | 748.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 753.55 | 746.87 | 748.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 755.10 | 746.87 | 748.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 755.00 | 748.49 | 749.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 757.95 | 748.49 | 749.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 754.95 | 749.78 | 749.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 757.05 | 752.52 | 751.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 754.90 | 756.16 | 754.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 752.00 | 755.32 | 753.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 752.00 | 755.32 | 753.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 745.80 | 753.42 | 753.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 745.80 | 753.42 | 753.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 736.25 | 749.99 | 751.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 733.80 | 740.88 | 743.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 743.35 | 740.99 | 742.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 745.40 | 741.88 | 742.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 746.40 | 741.88 | 742.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 743.00 | 743.20 | 743.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 749.60 | 743.20 | 743.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 749.45 | 744.45 | 743.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 753.35 | 747.40 | 745.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 758.00 | 758.49 | 752.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 758.00 | 758.49 | 752.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 751.30 | 756.29 | 752.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 751.30 | 756.29 | 752.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 749.35 | 754.90 | 752.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 749.35 | 754.90 | 752.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 747.55 | 750.85 | 751.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 745.10 | 748.16 | 749.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 747.25 | 745.50 | 747.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 747.25 | 745.50 | 747.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 747.05 | 745.81 | 747.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 747.70 | 745.81 | 747.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 744.40 | 745.53 | 747.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 741.80 | 745.53 | 747.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 742.30 | 745.31 | 747.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 743.40 | 744.35 | 746.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 742.95 | 743.61 | 745.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 749.15 | 744.72 | 745.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 749.15 | 744.72 | 745.77 | SL hit (close>static) qty=1.00 sl=747.35 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 749.20 | 746.81 | 746.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 772.25 | 751.90 | 748.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 780.00 | 780.12 | 773.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 779.95 | 780.12 | 773.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 775.70 | 778.84 | 774.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 775.70 | 778.84 | 774.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 773.05 | 777.69 | 774.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 778.95 | 777.69 | 774.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 771.90 | 774.76 | 774.34 | SL hit (close<static) qty=1.00 sl=773.05 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 769.65 | 774.42 | 774.45 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 779.90 | 775.31 | 774.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 782.90 | 777.28 | 775.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 790.75 | 792.52 | 787.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 790.75 | 792.52 | 787.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 788.35 | 791.68 | 787.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 788.35 | 791.68 | 787.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 799.25 | 803.68 | 800.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 799.25 | 803.68 | 800.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 805.70 | 804.08 | 800.77 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 792.40 | 799.92 | 800.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 786.85 | 793.44 | 796.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 775.60 | 774.49 | 780.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 775.95 | 771.83 | 775.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 773.00 | 771.83 | 775.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 779.40 | 773.34 | 775.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 779.30 | 773.34 | 775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 782.75 | 775.22 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 782.75 | 775.22 | 776.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 780.90 | 777.18 | 776.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 791.85 | 780.11 | 778.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 781.05 | 783.62 | 780.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 787.55 | 784.40 | 781.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 784.35 | 784.40 | 781.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 785.00 | 788.05 | 785.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 773.05 | 788.05 | 785.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 763.70 | 783.18 | 783.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 747.35 | 771.62 | 778.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 14:15:00 | 700.35 | 699.36 | 715.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:45:00 | 700.45 | 699.36 | 715.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 716.00 | 705.47 | 709.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 716.00 | 705.47 | 709.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 718.25 | 708.03 | 710.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 721.90 | 708.03 | 710.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 716.70 | 712.49 | 712.33 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 711.40 | 712.32 | 712.39 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 715.80 | 713.02 | 712.70 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 705.40 | 711.82 | 712.23 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 717.50 | 712.29 | 712.15 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 709.55 | 711.75 | 712.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 702.10 | 706.68 | 709.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 686.20 | 680.81 | 689.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 686.20 | 680.81 | 689.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 697.20 | 684.09 | 690.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 697.20 | 684.09 | 690.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 697.15 | 686.70 | 690.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 737.50 | 686.70 | 690.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 738.50 | 697.06 | 695.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 757.70 | 734.46 | 718.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 739.55 | 749.30 | 736.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 739.55 | 749.30 | 736.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 740.65 | 747.57 | 736.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 740.65 | 747.57 | 736.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 741.00 | 745.22 | 741.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 741.00 | 745.22 | 741.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 738.55 | 743.89 | 740.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 738.55 | 743.89 | 740.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 740.50 | 743.21 | 740.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 742.45 | 741.07 | 740.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:45:00 | 742.50 | 742.33 | 741.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 744.15 | 742.66 | 741.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 742.30 | 747.51 | 746.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 742.35 | 746.48 | 746.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 741.45 | 746.48 | 746.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 740.65 | 745.31 | 745.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 740.65 | 745.31 | 745.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 732.70 | 742.53 | 744.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 733.85 | 731.07 | 735.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 733.85 | 731.07 | 735.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 735.55 | 731.96 | 735.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 735.00 | 731.96 | 735.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 733.50 | 732.27 | 735.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 735.60 | 732.27 | 735.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 735.75 | 733.47 | 735.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 735.20 | 733.47 | 735.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 737.20 | 734.21 | 735.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 736.65 | 734.21 | 735.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 741.35 | 736.59 | 736.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 749.65 | 740.02 | 737.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 752.50 | 756.58 | 750.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:00:00 | 752.50 | 756.58 | 750.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 752.30 | 755.73 | 751.08 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 657.25 | 734.23 | 742.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 644.00 | 703.12 | 726.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 634.30 | 629.37 | 645.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:00:00 | 634.30 | 629.37 | 645.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 639.75 | 636.46 | 640.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 639.75 | 636.46 | 640.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 638.50 | 636.87 | 640.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 637.10 | 636.87 | 640.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 641.25 | 637.75 | 640.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 641.25 | 637.75 | 640.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 636.80 | 637.56 | 640.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 641.75 | 637.56 | 640.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 621.90 | 616.10 | 622.14 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 631.40 | 623.58 | 623.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 633.55 | 625.57 | 624.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 628.85 | 629.80 | 627.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 626.20 | 629.08 | 627.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 610.80 | 629.08 | 627.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 611.30 | 625.53 | 625.79 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 629.70 | 625.02 | 624.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 631.25 | 627.22 | 625.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 628.20 | 628.94 | 627.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 624.60 | 628.07 | 627.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 624.60 | 628.07 | 627.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 624.00 | 627.26 | 626.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 621.30 | 627.26 | 626.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 621.40 | 626.08 | 626.38 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 629.10 | 626.69 | 626.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 631.85 | 627.72 | 627.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 626.30 | 628.29 | 627.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 611.00 | 624.83 | 626.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 609.90 | 619.57 | 623.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 609.05 | 608.66 | 613.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 611.50 | 608.66 | 613.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 612.40 | 609.34 | 613.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 612.40 | 609.34 | 613.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 615.75 | 610.62 | 613.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 615.75 | 610.62 | 613.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 616.30 | 611.76 | 613.66 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 619.65 | 615.18 | 614.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 627.45 | 619.73 | 617.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 618.40 | 624.14 | 620.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 621.25 | 623.57 | 620.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 624.10 | 623.67 | 620.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 618.60 | 622.66 | 620.73 | SL hit (close<static) qty=1.00 sl=619.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 612.55 | 619.16 | 619.39 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 632.00 | 620.60 | 619.93 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 607.50 | 620.65 | 621.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 603.60 | 615.85 | 618.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 620.25 | 615.04 | 615.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 629.70 | 618.92 | 616.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 622.95 | 623.56 | 620.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 628.40 | 623.56 | 620.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 613.00 | 621.45 | 619.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 613.00 | 621.45 | 619.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 606.65 | 618.49 | 618.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 602.70 | 615.33 | 617.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 594.40 | 583.05 | 593.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 591.70 | 584.78 | 593.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 579.80 | 593.62 | 595.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 588.85 | 588.77 | 591.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 588.80 | 589.24 | 591.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 589.55 | 589.79 | 591.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 594.95 | 590.82 | 591.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 596.15 | 590.82 | 591.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 598.95 | 592.45 | 592.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 598.95 | 592.45 | 592.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 605.80 | 596.53 | 594.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 634.65 | 639.87 | 636.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 12:15:00 | 654.65 | 659.14 | 659.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 654.65 | 659.14 | 659.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 653.65 | 658.05 | 658.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 635.95 | 635.37 | 641.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:15:00 | 641.70 | 635.37 | 641.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 641.60 | 636.61 | 641.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 641.05 | 636.61 | 641.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 643.15 | 637.92 | 641.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 643.10 | 637.92 | 641.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 641.40 | 638.62 | 641.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 643.05 | 638.62 | 641.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 641.50 | 639.19 | 641.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 641.25 | 639.19 | 641.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 640.30 | 639.41 | 641.50 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 650.90 | 642.72 | 642.54 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 633.90 | 643.08 | 643.69 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 654.85 | 643.36 | 642.78 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 642.30 | 642.95 | 642.99 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 647.85 | 643.80 | 643.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 654.65 | 647.30 | 645.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 652.85 | 653.21 | 649.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 652.85 | 653.21 | 649.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 653.75 | 653.20 | 650.15 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 644.50 | 649.10 | 649.27 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 493.05 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-05-17 11:00:00 | 490.89 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-05-17 14:30:00 | 490.84 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-05-18 09:15:00 | 491.37 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-05-22 09:15:00 | 495.16 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-05-22 10:00:00 | 492.91 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-06-13 12:15:00 | 537.18 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-06-19 09:15:00 | 538.28 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-06-19 11:45:00 | 535.21 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-06-19 14:30:00 | 534.78 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2024-06-20 10:30:00 | 544.56 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-21 10:00:00 | 543.79 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-06-21 11:00:00 | 543.60 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-21 15:00:00 | 543.41 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-06-26 10:30:00 | 549.36 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-06-26 13:30:00 | 548.50 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-06-26 15:00:00 | 548.59 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-06-27 09:45:00 | 548.97 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-05 11:45:00 | 549.41 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-05 12:30:00 | 549.89 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-05 15:15:00 | 549.45 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-07-10 10:15:00 | 529.50 | 2024-07-12 12:15:00 | 541.88 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-07-10 12:15:00 | 530.41 | 2024-07-12 12:15:00 | 541.88 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-07-22 15:00:00 | 522.98 | 2024-07-26 12:15:00 | 521.30 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-07-23 11:15:00 | 522.12 | 2024-07-26 12:15:00 | 521.30 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-08-06 12:15:00 | 522.79 | 2024-08-07 15:15:00 | 526.14 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-12 12:30:00 | 536.36 | 2024-08-14 09:15:00 | 523.89 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-26 09:15:00 | 551.52 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-08-26 12:00:00 | 551.09 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-26 12:30:00 | 551.76 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-09-20 13:15:00 | 568.11 | 2024-09-24 10:15:00 | 581.25 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-09-20 14:15:00 | 566.72 | 2024-09-24 10:15:00 | 581.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-10-10 11:30:00 | 561.64 | 2024-10-17 11:15:00 | 533.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 12:45:00 | 561.64 | 2024-10-17 11:15:00 | 533.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-11 10:15:00 | 561.92 | 2024-10-17 11:15:00 | 533.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 11:30:00 | 561.64 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-10 12:45:00 | 561.64 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-11 10:15:00 | 561.92 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.12% |
| BUY | retest2 | 2024-11-04 15:00:00 | 530.65 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-11-05 09:15:00 | 533.77 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-11-05 12:30:00 | 530.32 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-11-13 09:15:00 | 500.34 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-11-14 12:15:00 | 500.25 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-18 09:15:00 | 498.52 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-12-06 10:45:00 | 565.30 | 2024-12-09 09:15:00 | 552.35 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-12-16 15:15:00 | 547.00 | 2024-12-17 09:15:00 | 550.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-12-17 11:45:00 | 544.70 | 2024-12-19 15:15:00 | 517.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 544.70 | 2024-12-23 11:15:00 | 509.80 | STOP_HIT | 0.50 | 6.41% |
| BUY | retest2 | 2025-01-07 10:15:00 | 542.05 | 2025-01-13 11:15:00 | 537.05 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-08 09:30:00 | 540.30 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-08 15:00:00 | 540.30 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-10 10:30:00 | 540.65 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-01-13 10:15:00 | 546.95 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-01-14 12:15:00 | 537.40 | 2025-01-14 13:15:00 | 542.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-01-16 09:15:00 | 550.60 | 2025-01-20 09:15:00 | 547.65 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-01-17 09:15:00 | 550.00 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-20 10:15:00 | 548.80 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-21 10:45:00 | 549.00 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-01-21 11:45:00 | 551.25 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2025-01-31 09:15:00 | 570.95 | 2025-01-31 14:15:00 | 599.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-01-31 09:15:00 | 570.95 | 2025-02-01 15:15:00 | 599.60 | STOP_HIT | 0.50 | 5.02% |
| BUY | retest2 | 2025-02-06 15:00:00 | 644.65 | 2025-02-10 12:15:00 | 633.70 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-02-07 11:30:00 | 644.85 | 2025-02-10 12:15:00 | 633.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-19 10:15:00 | 643.70 | 2025-02-27 10:15:00 | 641.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-02-21 10:45:00 | 645.00 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-02-21 15:00:00 | 648.40 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-02-24 11:30:00 | 644.45 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-02-25 09:15:00 | 651.60 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-02-28 09:15:00 | 625.00 | 2025-03-05 13:15:00 | 629.25 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-03-13 11:15:00 | 606.70 | 2025-03-17 10:15:00 | 616.05 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-13 14:00:00 | 606.50 | 2025-03-17 10:15:00 | 616.05 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-04-03 11:45:00 | 646.10 | 2025-04-04 13:15:00 | 640.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-04 10:30:00 | 648.05 | 2025-04-04 13:15:00 | 640.45 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-04-17 10:30:00 | 664.15 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest1 | 2025-04-17 12:30:00 | 660.75 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest1 | 2025-04-21 09:15:00 | 661.20 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-04-21 14:30:00 | 668.65 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-23 09:15:00 | 671.20 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-23 10:15:00 | 668.80 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-04-23 10:45:00 | 668.00 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-22 09:45:00 | 635.80 | 2025-05-28 10:15:00 | 637.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-30 10:15:00 | 623.05 | 2025-06-02 11:15:00 | 633.55 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-06 14:00:00 | 644.80 | 2025-06-09 11:15:00 | 639.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-12 12:00:00 | 635.45 | 2025-06-16 11:15:00 | 639.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-06-12 13:00:00 | 635.85 | 2025-06-16 11:15:00 | 639.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-14 10:45:00 | 652.20 | 2025-07-14 15:15:00 | 661.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-21 10:15:00 | 697.30 | 2025-07-30 11:15:00 | 719.45 | STOP_HIT | 1.00 | 3.18% |
| SELL | retest2 | 2025-08-04 12:00:00 | 705.25 | 2025-08-04 13:15:00 | 716.15 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-05 10:00:00 | 703.60 | 2025-08-05 10:15:00 | 711.35 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-08 09:15:00 | 702.10 | 2025-08-18 10:15:00 | 697.35 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-08-29 15:00:00 | 715.60 | 2025-09-01 09:15:00 | 730.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-09-15 12:15:00 | 688.95 | 2025-09-16 11:15:00 | 700.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-16 09:15:00 | 687.35 | 2025-09-16 11:15:00 | 700.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-18 14:30:00 | 699.35 | 2025-09-18 15:15:00 | 698.20 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-09-30 10:30:00 | 653.85 | 2025-10-01 13:15:00 | 661.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-01 10:00:00 | 654.90 | 2025-10-01 13:15:00 | 661.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-08 12:15:00 | 681.05 | 2025-10-08 13:15:00 | 676.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-08 13:30:00 | 682.55 | 2025-10-08 14:15:00 | 674.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-14 11:15:00 | 669.50 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-14 11:45:00 | 669.65 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-14 15:00:00 | 669.80 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-06 14:00:00 | 726.25 | 2025-11-18 15:15:00 | 760.00 | STOP_HIT | 1.00 | 4.65% |
| BUY | retest2 | 2025-11-07 09:45:00 | 727.60 | 2025-11-18 15:15:00 | 760.00 | STOP_HIT | 1.00 | 4.45% |
| SELL | retest2 | 2025-11-20 14:45:00 | 752.30 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-21 09:15:00 | 750.25 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-24 10:45:00 | 750.25 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-25 12:15:00 | 750.50 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-25 13:30:00 | 747.00 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-11-25 14:30:00 | 746.95 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-18 14:15:00 | 741.80 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-18 15:15:00 | 742.30 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-19 10:45:00 | 743.40 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-19 12:30:00 | 742.95 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-26 09:15:00 | 778.95 | 2025-12-26 15:15:00 | 771.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-29 09:15:00 | 776.55 | 2025-12-29 12:15:00 | 769.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-02-09 11:00:00 | 742.45 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-02-09 13:45:00 | 742.50 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-09 14:45:00 | 744.15 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-12 09:30:00 | 742.30 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-03-19 12:00:00 | 624.10 | 2026-03-19 12:15:00 | 618.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-02 09:15:00 | 579.80 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-02 14:30:00 | 588.85 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-06 09:15:00 | 588.80 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-06 09:45:00 | 589.55 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-13 10:45:00 | 634.65 | 2026-04-21 12:15:00 | 654.65 | STOP_HIT | 1.00 | 3.15% |
