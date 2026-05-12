# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 53 |
| ALERT2 | 52 |
| ALERT2_SKIP | 30 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 66 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 29 / 50
- **Target hits / Stop hits / Partials:** 1 / 69 / 9
- **Avg / median % per leg:** 0.27% / -0.77%
- **Sum % (uncompounded):** 20.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 7 | 31.8% | 1 | 21 | 0 | -0.49% | -10.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.51% | -5.0% |
| BUY @ 3rd Alert (retest2) | 20 | 7 | 35.0% | 1 | 19 | 0 | -0.28% | -5.7% |
| SELL (all) | 57 | 22 | 38.6% | 0 | 48 | 9 | 0.56% | 31.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.05% | -3.1% |
| SELL @ 3rd Alert (retest2) | 54 | 22 | 40.7% | 0 | 45 | 9 | 0.64% | 34.8% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.63% | -8.2% |
| retest2 (combined) | 74 | 29 | 39.2% | 1 | 64 | 9 | 0.39% | 29.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 548.05 | 537.36 | 536.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 553.80 | 547.86 | 544.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 551.15 | 553.63 | 549.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:15:00 | 549.00 | 553.63 | 549.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 536.50 | 550.21 | 548.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 536.50 | 550.21 | 548.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 538.80 | 547.93 | 547.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 534.35 | 547.93 | 547.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 541.00 | 546.54 | 546.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 530.80 | 542.07 | 544.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 537.75 | 534.33 | 536.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 536.00 | 534.67 | 536.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 533.50 | 534.67 | 536.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 532.60 | 535.15 | 535.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:00:00 | 534.80 | 534.84 | 535.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 541.00 | 536.71 | 536.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 541.00 | 536.71 | 536.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 549.10 | 543.01 | 540.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 543.05 | 543.37 | 540.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 539.10 | 542.51 | 540.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 539.10 | 542.51 | 540.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 539.10 | 542.51 | 540.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 538.00 | 541.61 | 540.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:30:00 | 539.45 | 541.61 | 540.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 544.15 | 545.76 | 542.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 544.15 | 545.76 | 542.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 548.10 | 546.23 | 543.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 545.30 | 546.23 | 543.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 543.40 | 546.07 | 543.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 543.40 | 546.07 | 543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 545.45 | 545.95 | 544.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:45:00 | 542.50 | 545.95 | 544.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 559.45 | 548.65 | 545.40 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 540.65 | 548.24 | 549.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 538.90 | 544.05 | 546.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 547.40 | 544.08 | 546.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 545.20 | 544.30 | 545.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 543.65 | 544.30 | 545.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 543.00 | 544.53 | 545.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 548.50 | 545.40 | 546.05 | SL hit (close>static) qty=1.00 sl=547.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 551.55 | 546.84 | 546.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 15:15:00 | 554.95 | 551.18 | 549.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 565.90 | 570.09 | 561.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 565.90 | 570.09 | 561.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 565.80 | 569.23 | 562.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 579.60 | 569.23 | 562.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:45:00 | 574.95 | 575.02 | 571.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 568.85 | 572.82 | 571.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:45:00 | 568.80 | 572.07 | 571.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 567.00 | 571.06 | 571.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 567.00 | 571.06 | 571.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 561.80 | 568.46 | 569.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 539.95 | 539.52 | 544.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 537.60 | 539.52 | 544.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 544.80 | 540.61 | 544.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 544.80 | 540.61 | 544.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 540.15 | 540.51 | 543.74 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 548.85 | 545.29 | 544.83 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 535.75 | 543.10 | 544.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 533.50 | 541.18 | 543.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 520.00 | 509.44 | 516.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 513.15 | 510.18 | 516.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 509.40 | 510.03 | 515.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 509.35 | 506.90 | 511.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 499.25 | 497.89 | 497.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 499.25 | 497.89 | 497.71 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 494.10 | 497.11 | 497.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 489.00 | 494.56 | 496.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 490.70 | 490.65 | 492.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:00:00 | 487.75 | 490.07 | 492.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 12:00:00 | 487.30 | 489.52 | 491.98 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 14:00:00 | 488.00 | 488.39 | 490.99 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 492.80 | 489.27 | 491.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 492.80 | 489.27 | 491.15 | SL hit (close>ema400) qty=1.00 sl=491.15 alert=retest1 |

### Cycle 11 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 499.90 | 492.34 | 492.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 502.55 | 495.71 | 493.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 499.80 | 500.10 | 497.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:30:00 | 500.25 | 500.10 | 497.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 491.60 | 498.43 | 497.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 491.60 | 498.43 | 497.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 490.60 | 496.86 | 497.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 490.00 | 495.49 | 496.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 482.30 | 485.16 | 487.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 482.95 | 484.03 | 486.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 487.50 | 485.42 | 485.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 15:15:00 | 487.50 | 485.42 | 485.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 518.20 | 491.98 | 488.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 13:15:00 | 506.05 | 507.81 | 502.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:45:00 | 505.45 | 507.81 | 502.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 502.25 | 506.60 | 503.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 502.25 | 506.60 | 503.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 503.05 | 505.89 | 503.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 503.05 | 505.89 | 503.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 500.55 | 504.46 | 503.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 500.55 | 504.46 | 503.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 501.80 | 503.93 | 503.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 497.40 | 503.93 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 497.90 | 502.73 | 502.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 492.55 | 499.73 | 501.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 496.20 | 494.53 | 496.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 494.95 | 494.61 | 496.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 491.00 | 493.89 | 496.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 466.45 | 477.73 | 483.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 477.70 | 476.97 | 481.51 | SL hit (close>ema200) qty=0.50 sl=476.97 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 491.10 | 482.06 | 482.02 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 481.10 | 482.00 | 482.08 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 494.25 | 484.37 | 483.13 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 478.80 | 482.97 | 483.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 472.50 | 480.88 | 482.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 483.00 | 478.63 | 478.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 492.95 | 481.49 | 479.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 484.20 | 489.26 | 484.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 484.00 | 488.20 | 484.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 480.80 | 486.72 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 483.00 | 485.98 | 484.30 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 478.85 | 483.05 | 483.25 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 495.45 | 485.46 | 484.29 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 485.15 | 488.14 | 488.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 483.20 | 487.15 | 487.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 11:15:00 | 481.10 | 480.50 | 483.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 481.10 | 480.50 | 483.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 480.95 | 480.42 | 482.05 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 486.70 | 482.00 | 481.93 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 482.35 | 482.87 | 482.90 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 484.50 | 483.20 | 483.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 486.60 | 483.88 | 483.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 485.75 | 486.61 | 485.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 485.15 | 486.32 | 485.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:15:00 | 484.65 | 486.32 | 485.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 482.15 | 485.49 | 485.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 482.15 | 485.49 | 485.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 482.05 | 484.80 | 484.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 486.35 | 484.80 | 484.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 497.30 | 487.30 | 485.93 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 483.40 | 488.57 | 488.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 476.00 | 483.17 | 485.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 480.65 | 480.01 | 482.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 482.75 | 480.01 | 482.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 481.80 | 480.37 | 482.22 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 482.85 | 482.56 | 482.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 485.10 | 483.27 | 482.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 492.25 | 492.56 | 488.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:30:00 | 492.80 | 492.56 | 488.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 495.25 | 492.78 | 489.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 499.65 | 494.16 | 492.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 488.45 | 491.88 | 491.74 | SL hit (close<static) qty=1.00 sl=489.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 487.30 | 490.97 | 491.34 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 506.55 | 494.00 | 492.49 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 495.65 | 498.59 | 498.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 15:15:00 | 495.40 | 497.12 | 498.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 499.45 | 497.56 | 498.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 494.70 | 496.99 | 497.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 493.35 | 496.57 | 497.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 493.45 | 495.02 | 496.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 491.80 | 493.88 | 495.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 493.95 | 493.31 | 495.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 493.20 | 492.79 | 494.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 494.15 | 492.79 | 494.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 493.60 | 492.95 | 494.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 493.20 | 492.95 | 494.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 494.70 | 493.43 | 494.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 494.70 | 493.43 | 494.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 493.75 | 493.49 | 494.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:15:00 | 494.25 | 493.49 | 494.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 496.00 | 493.99 | 494.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 496.00 | 493.99 | 494.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 495.00 | 494.20 | 494.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 496.00 | 494.20 | 494.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 495.00 | 494.36 | 494.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 495.30 | 494.54 | 494.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 495.30 | 494.54 | 494.50 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 493.05 | 494.29 | 494.39 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 500.00 | 494.77 | 494.52 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 490.60 | 494.23 | 494.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 489.40 | 492.26 | 493.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 481.80 | 481.71 | 485.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 481.80 | 481.71 | 485.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 474.50 | 474.16 | 477.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 474.90 | 474.16 | 477.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 472.50 | 472.74 | 474.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 463.45 | 470.13 | 472.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 472.50 | 470.33 | 470.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 472.50 | 470.33 | 470.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 473.00 | 470.87 | 470.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 470.75 | 471.41 | 470.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 470.10 | 471.15 | 470.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:30:00 | 471.10 | 471.15 | 470.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 470.45 | 471.01 | 470.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:00:00 | 472.35 | 471.28 | 470.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 468.60 | 471.16 | 471.02 | SL hit (close<static) qty=1.00 sl=469.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 467.95 | 470.49 | 470.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 14:15:00 | 465.95 | 469.07 | 470.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 470.25 | 467.65 | 468.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 469.15 | 467.95 | 468.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 469.80 | 467.95 | 468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 471.80 | 468.91 | 469.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 471.80 | 468.91 | 469.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 469.50 | 469.03 | 469.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 468.50 | 468.52 | 468.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 464.60 | 458.08 | 457.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 464.60 | 458.08 | 457.70 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 451.55 | 457.68 | 457.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 450.70 | 455.32 | 456.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 457.55 | 455.76 | 456.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 457.40 | 456.09 | 456.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 459.10 | 456.09 | 456.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 460.60 | 457.83 | 457.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 456.30 | 457.99 | 458.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 12:15:00 | 453.60 | 457.11 | 457.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 455.85 | 457.40 | 457.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 455.90 | 453.14 | 453.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 455.85 | 453.14 | 453.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 457.80 | 454.64 | 454.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 457.80 | 454.64 | 454.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 460.55 | 455.82 | 454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 452.75 | 455.11 | 455.34 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 456.90 | 455.05 | 454.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 459.00 | 456.25 | 455.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 458.05 | 458.73 | 457.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 458.25 | 458.63 | 457.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 459.20 | 458.71 | 457.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 458.75 | 458.66 | 457.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 455.25 | 457.98 | 457.58 | SL hit (close<static) qty=1.00 sl=457.40 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 453.00 | 456.99 | 457.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 451.90 | 454.63 | 455.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 417.35 | 415.87 | 426.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 419.90 | 415.87 | 426.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 420.00 | 417.67 | 424.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 417.80 | 418.51 | 423.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 14:15:00 | 396.91 | 403.72 | 407.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 404.00 | 402.57 | 406.59 | SL hit (close>ema200) qty=0.50 sl=402.57 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 378.80 | 376.59 | 376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 384.00 | 379.26 | 377.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 15:15:00 | 397.15 | 397.54 | 391.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 406.45 | 397.54 | 391.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 13:15:00 | 400.55 | 400.01 | 395.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 395.90 | 399.02 | 395.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 394.00 | 399.02 | 395.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 393.35 | 397.89 | 395.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 393.35 | 397.89 | 395.31 | SL hit (close<ema400) qty=1.00 sl=395.31 alert=retest1 |

### Cycle 46 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 382.70 | 392.62 | 393.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 380.60 | 390.22 | 392.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 369.25 | 367.84 | 371.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 369.25 | 367.84 | 371.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 371.80 | 368.83 | 371.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 362.85 | 368.83 | 371.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 344.71 | 351.78 | 359.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 359.20 | 351.94 | 358.43 | SL hit (close>ema200) qty=0.50 sl=351.94 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 365.15 | 360.77 | 360.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 376.50 | 367.30 | 365.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 373.15 | 377.33 | 374.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 368.00 | 375.46 | 374.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 368.00 | 375.46 | 374.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 361.65 | 371.32 | 372.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 350.85 | 367.23 | 370.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 15:15:00 | 349.10 | 348.46 | 353.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:15:00 | 352.45 | 348.46 | 353.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 351.90 | 349.15 | 353.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 347.45 | 349.01 | 352.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 346.85 | 348.71 | 351.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 347.00 | 348.51 | 351.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:45:00 | 347.10 | 347.62 | 349.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 349.70 | 347.93 | 349.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 351.30 | 347.93 | 349.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 350.00 | 348.35 | 349.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:15:00 | 351.55 | 348.35 | 349.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 351.10 | 348.90 | 349.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 351.50 | 348.90 | 349.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 350.85 | 349.40 | 349.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 350.85 | 349.40 | 349.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 349.95 | 349.51 | 349.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:15:00 | 349.00 | 349.51 | 349.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 349.00 | 349.69 | 349.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 349.20 | 349.18 | 349.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 351.90 | 349.72 | 349.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 15:15:00 | 351.90 | 349.72 | 349.63 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 346.05 | 348.99 | 349.31 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 361.00 | 351.64 | 350.47 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 351.95 | 352.21 | 352.22 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 354.50 | 352.20 | 352.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 358.05 | 354.27 | 353.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 354.90 | 365.00 | 361.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 352.95 | 362.59 | 360.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 352.95 | 362.59 | 360.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 353.40 | 359.06 | 359.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 351.55 | 355.33 | 357.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 370.95 | 357.13 | 357.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 13:15:00 | 373.35 | 360.37 | 359.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 377.55 | 363.81 | 360.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 366.35 | 368.85 | 365.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:00:00 | 366.35 | 368.85 | 365.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 365.50 | 367.69 | 365.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 11:45:00 | 370.40 | 367.78 | 366.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 369.35 | 368.03 | 366.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 369.55 | 368.03 | 366.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 356.30 | 364.71 | 365.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 356.30 | 364.71 | 365.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 351.85 | 362.14 | 364.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 352.60 | 352.52 | 355.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 349.20 | 352.52 | 355.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 349.00 | 348.74 | 351.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 345.35 | 348.81 | 350.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 346.70 | 348.53 | 350.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 347.05 | 348.30 | 349.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 346.00 | 347.84 | 349.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 349.75 | 348.22 | 349.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 349.75 | 348.22 | 349.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 348.15 | 348.21 | 349.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 346.00 | 347.77 | 349.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.08 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 329.36 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 329.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 330.55 | 328.05 | 333.07 | SL hit (close>ema200) qty=0.50 sl=328.05 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 332.70 | 330.49 | 330.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 335.15 | 331.83 | 330.93 | Break + close above crossover candle high |

### Cycle 58 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 323.70 | 330.21 | 330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 321.75 | 328.51 | 329.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 328.70 | 327.56 | 328.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 329.30 | 327.91 | 328.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 329.00 | 327.91 | 328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 327.50 | 327.83 | 328.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 325.00 | 327.83 | 328.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 332.70 | 328.80 | 329.02 | SL hit (close>static) qty=1.00 sl=329.80 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 335.00 | 330.04 | 329.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 337.10 | 333.57 | 331.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 331.00 | 337.81 | 335.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 329.30 | 336.11 | 334.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 329.30 | 336.11 | 334.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 323.55 | 333.59 | 333.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 13:15:00 | 322.40 | 327.45 | 329.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 320.70 | 327.48 | 329.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 331.25 | 327.31 | 327.54 | SL hit (close>static) qty=1.00 sl=329.65 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 333.30 | 328.51 | 328.07 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 327.75 | 328.65 | 328.73 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 349.05 | 332.59 | 330.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 354.25 | 347.36 | 340.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 348.50 | 350.53 | 345.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 352.15 | 350.53 | 345.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 355.65 | 357.92 | 355.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 355.80 | 357.92 | 355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 354.05 | 357.14 | 355.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 347.35 | 357.14 | 355.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 346.35 | 354.99 | 354.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 345.05 | 354.99 | 354.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 346.50 | 353.29 | 353.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 343.45 | 348.93 | 351.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 342.25 | 340.91 | 344.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 342.25 | 340.91 | 344.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 341.05 | 340.94 | 344.30 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 345.85 | 344.69 | 344.68 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 343.05 | 344.66 | 344.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 341.05 | 343.94 | 344.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 341.75 | 340.59 | 342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 341.95 | 340.86 | 342.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 343.50 | 340.86 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 342.25 | 341.14 | 342.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 342.50 | 341.14 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 343.00 | 341.51 | 342.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 342.25 | 341.51 | 342.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 343.00 | 341.81 | 342.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 342.85 | 341.81 | 342.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 343.15 | 342.20 | 342.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 343.15 | 342.20 | 342.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 343.50 | 342.46 | 342.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 343.50 | 342.46 | 342.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 343.90 | 342.74 | 342.64 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 340.05 | 342.13 | 342.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 335.65 | 340.83 | 341.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 333.55 | 331.19 | 334.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 333.85 | 332.15 | 333.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 334.00 | 332.15 | 333.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 332.50 | 332.22 | 333.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:30:00 | 334.10 | 332.22 | 333.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 333.00 | 329.22 | 330.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 333.00 | 329.22 | 330.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 333.30 | 330.03 | 330.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 333.15 | 330.03 | 330.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 328.00 | 328.17 | 329.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 320.85 | 328.17 | 329.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 329.05 | 325.14 | 324.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 329.05 | 325.14 | 324.68 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 309.50 | 321.82 | 323.44 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 325.05 | 322.85 | 322.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 326.80 | 324.39 | 323.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 322.00 | 327.55 | 326.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 326.10 | 327.26 | 326.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 327.35 | 327.44 | 326.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-16 09:15:00 | 360.09 | 344.97 | 339.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 342.10 | 348.95 | 348.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 339.55 | 345.85 | 347.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 343.80 | 341.84 | 344.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 344.00 | 342.27 | 344.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 343.70 | 342.27 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 344.45 | 342.71 | 344.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 344.45 | 342.71 | 344.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 344.00 | 342.97 | 344.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 344.00 | 342.97 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 344.35 | 343.24 | 344.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:30:00 | 344.10 | 343.24 | 344.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 345.30 | 343.66 | 344.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 333.15 | 343.66 | 344.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 343.50 | 339.82 | 339.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 343.50 | 339.82 | 339.72 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 11:15:00 | 337.25 | 339.68 | 339.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 12:15:00 | 336.00 | 338.94 | 339.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 329.90 | 328.04 | 332.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 329.90 | 328.04 | 332.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 332.70 | 328.98 | 332.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 332.70 | 328.98 | 332.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 330.00 | 329.18 | 332.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:45:00 | 328.00 | 329.36 | 332.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 328.20 | 329.36 | 332.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 328.85 | 329.53 | 331.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:15:00 | 325.85 | 329.53 | 331.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 337.75 | 327.98 | 329.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 337.75 | 327.98 | 329.81 | SL hit (close>static) qty=1.00 sl=333.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 13:15:00 | 321.00 | 319.05 | 319.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 323.00 | 320.16 | 319.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 315.50 | 319.22 | 319.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 10:15:00 | 316.35 | 318.65 | 318.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 12:15:00 | 313.80 | 317.39 | 318.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 317.90 | 316.93 | 317.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 317.25 | 316.99 | 317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 325.70 | 316.99 | 317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 328.60 | 319.31 | 318.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 15:15:00 | 332.20 | 326.23 | 323.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 339.55 | 340.14 | 335.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:00:00 | 339.55 | 340.14 | 335.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 343.00 | 343.57 | 341.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:45:00 | 346.75 | 343.07 | 341.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 339.75 | 342.37 | 341.97 | SL hit (close<static) qty=1.00 sl=340.05 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 339.70 | 341.58 | 341.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 339.35 | 341.13 | 341.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 341.30 | 340.85 | 341.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 342.95 | 341.27 | 341.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 343.45 | 341.27 | 341.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 343.30 | 341.68 | 341.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 15:15:00 | 344.80 | 342.30 | 341.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 340.35 | 342.03 | 341.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 342.60 | 342.15 | 341.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 342.30 | 342.15 | 341.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 343.65 | 342.45 | 342.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 347.40 | 342.45 | 342.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:00:00 | 345.35 | 345.24 | 343.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 345.65 | 345.22 | 343.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 345.15 | 345.08 | 344.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 346.00 | 345.26 | 344.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 346.65 | 345.54 | 344.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 349.15 | 345.75 | 344.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 350.60 | 354.82 | 355.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 350.60 | 354.82 | 355.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 342.70 | 352.39 | 353.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 348.30 | 346.14 | 348.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 348.30 | 346.14 | 348.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 346.60 | 346.23 | 348.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 349.60 | 346.23 | 348.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 346.75 | 345.82 | 347.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 350.05 | 345.82 | 347.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 345.50 | 344.44 | 345.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 343.30 | 344.24 | 345.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 343.90 | 344.79 | 345.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 347.50 | 345.33 | 345.76 | SL hit (close>static) qty=1.00 sl=347.10 alert=retest2 |

### Cycle 81 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 350.70 | 346.41 | 346.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 353.70 | 348.61 | 347.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 349.60 | 350.57 | 348.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 349.60 | 350.57 | 348.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 350.90 | 350.54 | 349.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 347.95 | 350.54 | 349.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 350.95 | 350.64 | 349.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 359.10 | 352.12 | 350.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 09:15:00 | 533.50 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-22 09:15:00 | 532.60 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-22 11:00:00 | 534.80 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-02 12:15:00 | 543.65 | 2025-06-02 14:15:00 | 548.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-02 13:15:00 | 543.00 | 2025-06-02 14:15:00 | 548.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-05 09:15:00 | 579.60 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-06-06 11:45:00 | 574.95 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-06-09 10:45:00 | 568.85 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-06-09 11:45:00 | 568.80 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-23 12:00:00 | 509.40 | 2025-07-03 13:15:00 | 499.25 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2025-06-24 10:00:00 | 509.35 | 2025-07-03 13:15:00 | 499.25 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest1 | 2025-07-08 11:00:00 | 487.75 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2025-07-08 12:00:00 | 487.30 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest1 | 2025-07-08 14:00:00 | 488.00 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-15 15:00:00 | 482.30 | 2025-07-17 15:15:00 | 487.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-16 12:30:00 | 482.95 | 2025-07-17 15:15:00 | 487.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-25 10:00:00 | 491.00 | 2025-07-29 09:15:00 | 466.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 10:00:00 | 491.00 | 2025-07-29 12:15:00 | 477.70 | STOP_HIT | 0.50 | 2.71% |
| BUY | retest2 | 2025-09-04 09:30:00 | 499.65 | 2025-09-05 09:15:00 | 488.45 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-12 13:15:00 | 493.35 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-09-12 14:45:00 | 493.45 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-15 09:45:00 | 491.80 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-29 15:00:00 | 463.45 | 2025-10-01 13:15:00 | 472.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-03 14:00:00 | 472.35 | 2025-10-06 09:15:00 | 468.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-06 11:15:00 | 472.10 | 2025-10-06 11:15:00 | 467.95 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-08 09:30:00 | 468.50 | 2025-10-13 11:15:00 | 464.60 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-10-17 09:15:00 | 455.85 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-20 13:30:00 | 455.90 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-20 14:15:00 | 455.85 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-31 12:30:00 | 459.20 | 2025-10-31 14:15:00 | 455.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-31 13:45:00 | 458.75 | 2025-10-31 14:15:00 | 455.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-10 15:15:00 | 417.80 | 2025-11-13 14:15:00 | 396.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 15:15:00 | 417.80 | 2025-11-14 09:15:00 | 404.00 | STOP_HIT | 0.50 | 3.30% |
| BUY | retest1 | 2025-12-01 09:15:00 | 406.45 | 2025-12-01 15:15:00 | 393.35 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest1 | 2025-12-01 13:15:00 | 400.55 | 2025-12-01 15:15:00 | 393.35 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-12-08 09:15:00 | 362.85 | 2025-12-09 09:15:00 | 344.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 362.85 | 2025-12-09 11:15:00 | 359.20 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2025-12-10 10:00:00 | 367.40 | 2025-12-10 10:15:00 | 365.15 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-12-22 12:30:00 | 347.45 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-22 13:45:00 | 346.85 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-22 15:15:00 | 347.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-23 12:45:00 | 347.10 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-24 15:15:00 | 349.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-26 11:15:00 | 349.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-26 15:00:00 | 349.20 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-09 11:45:00 | 370.40 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-01-09 12:45:00 | 369.35 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2026-01-09 13:15:00 | 369.55 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-01-19 09:15:00 | 345.35 | 2026-01-21 09:15:00 | 328.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 346.70 | 2026-01-21 09:15:00 | 329.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:15:00 | 347.05 | 2026-01-21 09:15:00 | 329.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 346.00 | 2026-01-21 09:15:00 | 328.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:00:00 | 346.00 | 2026-01-21 09:15:00 | 328.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 345.35 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2026-01-19 10:15:00 | 346.70 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-19 11:15:00 | 347.05 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-19 12:00:00 | 346.00 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-01-19 15:00:00 | 346.00 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-01-30 09:15:00 | 325.00 | 2026-01-30 09:15:00 | 332.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-04 09:15:00 | 320.70 | 2026-02-05 10:15:00 | 331.25 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-04 09:15:00 | 320.85 | 2026-03-06 09:15:00 | 329.05 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-12 11:45:00 | 327.35 | 2026-03-16 09:15:00 | 360.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 333.15 | 2026-03-25 09:15:00 | 343.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-03-30 09:45:00 | 328.00 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-03-30 10:15:00 | 328.20 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-03-30 13:45:00 | 328.85 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-03-30 14:15:00 | 325.85 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-01 15:15:00 | 326.00 | 2026-04-07 15:15:00 | 309.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 15:15:00 | 326.00 | 2026-04-08 09:15:00 | 320.80 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest2 | 2026-04-21 09:45:00 | 346.75 | 2026-04-21 13:15:00 | 339.75 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-23 13:15:00 | 347.40 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2026-04-24 10:00:00 | 345.35 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-04-24 10:45:00 | 345.65 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2026-04-24 12:30:00 | 345.15 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2026-04-24 15:00:00 | 346.65 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2026-04-27 09:15:00 | 349.15 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-05-06 10:45:00 | 343.30 | 2026-05-06 13:15:00 | 347.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-05-06 12:30:00 | 343.90 | 2026-05-06 13:15:00 | 347.50 | STOP_HIT | 1.00 | -1.05% |
