# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 955.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 143 |
| ALERT1 | 92 |
| ALERT2 | 89 |
| ALERT2_SKIP | 43 |
| ALERT3 | 220 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 111 |
| PARTIAL | 13 |
| TARGET_HIT | 15 |
| STOP_HIT | 97 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 79
- **Target hits / Stop hits / Partials:** 15 / 97 / 13
- **Avg / median % per leg:** 0.82% / -0.56%
- **Sum % (uncompounded):** 103.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 17 | 27.4% | 10 | 52 | 0 | 0.54% | 33.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 62 | 17 | 27.4% | 10 | 52 | 0 | 0.54% | 33.2% |
| SELL (all) | 63 | 29 | 46.0% | 5 | 45 | 13 | 1.11% | 69.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.22% | -2.2% |
| SELL @ 3rd Alert (retest2) | 62 | 29 | 46.8% | 5 | 44 | 13 | 1.16% | 72.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.22% | -2.2% |
| retest2 (combined) | 124 | 46 | 37.1% | 15 | 96 | 13 | 0.85% | 105.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 379.05 | 371.44 | 370.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 380.55 | 374.52 | 372.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 378.80 | 379.35 | 376.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 378.80 | 379.35 | 376.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 377.40 | 378.96 | 376.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 384.90 | 378.96 | 376.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 15:15:00 | 393.00 | 398.15 | 398.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 15:15:00 | 393.00 | 398.15 | 398.53 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 401.95 | 398.38 | 398.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 406.70 | 400.28 | 399.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 14:15:00 | 420.60 | 421.14 | 412.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 15:00:00 | 420.60 | 421.14 | 412.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 409.50 | 419.06 | 412.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 415.45 | 419.06 | 412.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 381.00 | 411.45 | 410.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 381.00 | 411.45 | 410.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 369.60 | 403.08 | 406.36 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 421.65 | 403.90 | 401.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 424.00 | 410.63 | 405.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 433.55 | 433.76 | 425.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 433.55 | 433.76 | 425.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 427.60 | 430.82 | 426.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 428.05 | 430.82 | 426.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 428.90 | 430.44 | 427.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 443.80 | 430.44 | 427.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 12:15:00 | 488.18 | 478.49 | 469.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 474.50 | 476.48 | 476.69 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 485.00 | 478.19 | 477.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 488.50 | 481.09 | 478.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 488.75 | 489.25 | 484.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 10:30:00 | 489.45 | 489.25 | 484.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 513.35 | 517.74 | 513.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 513.35 | 517.74 | 513.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 514.85 | 517.16 | 513.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:45:00 | 520.40 | 518.21 | 514.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-02 09:15:00 | 572.44 | 556.72 | 539.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 529.00 | 556.48 | 557.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 516.40 | 548.47 | 553.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 471.15 | 464.55 | 482.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:30:00 | 473.00 | 464.55 | 482.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 487.90 | 471.15 | 482.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 487.90 | 471.15 | 482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 494.10 | 475.74 | 483.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 494.10 | 475.74 | 483.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 486.60 | 485.59 | 486.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:45:00 | 486.55 | 485.59 | 486.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 491.20 | 486.71 | 486.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:45:00 | 492.50 | 486.71 | 486.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 489.65 | 487.30 | 487.17 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 484.75 | 486.79 | 486.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 14:15:00 | 482.00 | 485.83 | 486.50 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 492.25 | 486.98 | 486.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 471.60 | 486.09 | 487.44 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 12:15:00 | 497.35 | 488.98 | 488.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 508.20 | 496.44 | 492.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 507.00 | 508.08 | 502.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 10:30:00 | 509.25 | 508.08 | 502.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 511.65 | 508.80 | 503.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 509.00 | 508.80 | 503.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 493.00 | 509.31 | 506.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 493.00 | 509.31 | 506.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 495.10 | 506.47 | 505.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 11:45:00 | 500.80 | 506.47 | 505.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 500.25 | 504.67 | 505.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 11:15:00 | 500.25 | 504.67 | 505.08 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 510.75 | 504.31 | 504.29 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 500.60 | 504.05 | 504.26 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 507.95 | 504.34 | 504.32 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 502.75 | 505.54 | 505.72 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 509.40 | 505.27 | 505.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 527.35 | 510.98 | 507.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 521.00 | 521.02 | 515.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 519.00 | 521.02 | 515.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 520.00 | 520.81 | 515.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 530.00 | 517.46 | 516.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 513.85 | 519.69 | 517.83 | SL hit (close<static) qty=1.00 sl=514.55 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 512.95 | 516.66 | 516.76 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 520.70 | 516.70 | 516.66 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 503.90 | 515.59 | 516.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 490.45 | 504.41 | 510.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 476.00 | 475.48 | 484.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:15:00 | 478.75 | 475.48 | 484.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 493.00 | 480.16 | 485.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 525.40 | 480.16 | 485.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 526.45 | 489.42 | 489.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 534.10 | 498.36 | 493.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 555.20 | 555.25 | 544.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 555.20 | 555.25 | 544.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 549.00 | 554.04 | 547.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 549.00 | 554.04 | 547.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 543.00 | 551.83 | 546.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 543.00 | 551.83 | 546.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 544.00 | 550.26 | 546.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:15:00 | 539.80 | 550.26 | 546.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 534.60 | 547.13 | 545.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 534.60 | 547.13 | 545.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 557.30 | 548.51 | 546.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 563.25 | 550.89 | 547.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 13:15:00 | 562.45 | 552.73 | 548.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 542.05 | 550.40 | 550.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 14:15:00 | 542.05 | 550.40 | 550.41 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 566.60 | 552.76 | 551.43 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 550.20 | 555.86 | 556.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 14:15:00 | 546.50 | 553.99 | 555.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 15:15:00 | 548.85 | 548.72 | 551.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:15:00 | 544.75 | 548.72 | 551.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 556.85 | 550.35 | 551.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 556.85 | 550.35 | 551.72 | SL hit (close>ema400) qty=1.00 sl=551.72 alert=retest1 |

### Cycle 27 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 11:15:00 | 555.40 | 551.13 | 550.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 13:15:00 | 557.20 | 554.58 | 553.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 09:15:00 | 561.35 | 562.39 | 559.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 561.35 | 562.39 | 559.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 561.35 | 562.39 | 559.35 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 552.70 | 557.71 | 557.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 552.50 | 556.66 | 557.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 11:15:00 | 555.35 | 554.55 | 555.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 11:15:00 | 555.35 | 554.55 | 555.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 555.35 | 554.55 | 555.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 556.10 | 554.55 | 555.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 553.50 | 553.76 | 554.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 560.65 | 553.76 | 554.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 555.75 | 554.15 | 555.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 557.90 | 554.15 | 555.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 554.75 | 554.27 | 554.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:15:00 | 557.05 | 554.27 | 554.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 555.65 | 554.55 | 555.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:15:00 | 553.30 | 554.55 | 555.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 552.80 | 550.75 | 552.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 551.25 | 550.75 | 552.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 525.63 | 533.80 | 540.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 525.16 | 533.80 | 540.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 523.69 | 533.80 | 540.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-09 09:15:00 | 497.97 | 513.32 | 525.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 491.80 | 479.03 | 478.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 495.00 | 490.16 | 487.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 13:15:00 | 489.95 | 491.15 | 488.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 13:15:00 | 489.95 | 491.15 | 488.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 489.95 | 491.15 | 488.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 488.65 | 491.15 | 488.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 485.10 | 489.94 | 488.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 485.10 | 489.94 | 488.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 487.00 | 489.35 | 488.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 484.30 | 489.35 | 488.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 482.50 | 487.98 | 487.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 482.50 | 487.98 | 487.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 480.00 | 486.39 | 487.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 475.05 | 484.12 | 486.04 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 553.75 | 491.93 | 487.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 562.45 | 516.45 | 500.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 559.90 | 562.54 | 545.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 09:30:00 | 566.00 | 562.54 | 545.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 547.65 | 556.28 | 551.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 547.75 | 556.28 | 551.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 562.40 | 557.50 | 552.08 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 546.05 | 552.17 | 552.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 14:15:00 | 538.65 | 548.40 | 550.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 11:15:00 | 548.85 | 543.39 | 546.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 11:15:00 | 548.85 | 543.39 | 546.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 548.85 | 543.39 | 546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:45:00 | 546.85 | 543.39 | 546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 543.00 | 543.31 | 546.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 542.00 | 543.34 | 546.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 620.00 | 553.01 | 545.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 09:15:00 | 620.00 | 553.01 | 545.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 10:15:00 | 644.00 | 571.21 | 554.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 641.75 | 647.09 | 618.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 641.75 | 647.09 | 618.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 619.15 | 642.93 | 621.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 619.15 | 642.93 | 621.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 618.65 | 638.07 | 621.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:45:00 | 612.60 | 638.07 | 621.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 611.05 | 632.67 | 620.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:30:00 | 608.60 | 632.67 | 620.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 605.70 | 627.27 | 619.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:45:00 | 605.70 | 627.27 | 619.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 602.25 | 615.36 | 615.34 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 612.00 | 614.69 | 615.04 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 618.00 | 615.17 | 615.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 610.45 | 615.36 | 615.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 601.20 | 612.53 | 614.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 15:15:00 | 591.95 | 591.59 | 598.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:15:00 | 583.85 | 591.59 | 598.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 582.30 | 589.74 | 597.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:00:00 | 577.85 | 587.36 | 595.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:00:00 | 576.80 | 585.25 | 593.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:15:00 | 578.90 | 582.86 | 589.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 578.00 | 580.99 | 586.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 579.20 | 578.68 | 583.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 609.00 | 588.90 | 587.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 609.00 | 588.90 | 587.16 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 584.60 | 588.27 | 588.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 572.00 | 584.00 | 586.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 583.40 | 582.47 | 585.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 583.40 | 582.47 | 585.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 583.40 | 582.47 | 585.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 583.40 | 582.47 | 585.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 585.55 | 583.09 | 585.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 582.00 | 583.05 | 584.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 552.90 | 567.86 | 574.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 556.90 | 554.55 | 563.70 | SL hit (close>ema200) qty=0.50 sl=554.55 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 555.90 | 536.66 | 534.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 561.90 | 541.71 | 537.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 561.45 | 563.52 | 556.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 561.45 | 563.52 | 556.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 549.10 | 560.64 | 555.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:15:00 | 568.10 | 557.33 | 556.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 12:15:00 | 564.25 | 558.56 | 556.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 13:15:00 | 564.45 | 559.64 | 557.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 14:00:00 | 564.45 | 560.60 | 558.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 547.50 | 559.14 | 558.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 547.50 | 559.14 | 558.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 552.40 | 557.79 | 557.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-07 11:15:00 | 550.00 | 556.23 | 556.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 550.00 | 556.23 | 556.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 549.00 | 554.79 | 556.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 555.00 | 551.64 | 553.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 555.00 | 551.64 | 553.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 555.00 | 551.64 | 553.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 558.30 | 551.64 | 553.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 549.75 | 551.26 | 553.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 545.70 | 549.23 | 551.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:00:00 | 542.20 | 547.82 | 551.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 518.41 | 522.70 | 528.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 515.09 | 519.33 | 523.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 518.25 | 517.85 | 521.47 | SL hit (close>ema200) qty=0.50 sl=517.85 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 505.55 | 500.02 | 499.84 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 495.00 | 499.63 | 499.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 494.50 | 496.93 | 498.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 498.15 | 496.91 | 497.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 498.15 | 496.91 | 497.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 498.15 | 496.91 | 497.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 498.15 | 496.91 | 497.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 503.45 | 498.22 | 498.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 506.35 | 498.22 | 498.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 511.50 | 500.87 | 499.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 517.00 | 506.74 | 502.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 510.90 | 510.94 | 506.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:45:00 | 510.75 | 510.94 | 506.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 508.00 | 509.96 | 507.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 507.35 | 509.96 | 507.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 510.35 | 510.04 | 507.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:00:00 | 514.50 | 510.93 | 508.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:30:00 | 517.00 | 512.05 | 508.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 512.15 | 516.80 | 516.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 10:15:00 | 512.15 | 516.80 | 516.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 11:15:00 | 511.35 | 515.71 | 516.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 513.00 | 512.11 | 513.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:00:00 | 513.00 | 512.11 | 513.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 512.70 | 512.23 | 513.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 512.70 | 512.23 | 513.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 508.00 | 509.36 | 511.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 508.00 | 509.36 | 511.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 507.35 | 508.74 | 510.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 505.60 | 508.75 | 510.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 506.20 | 508.15 | 509.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 516.00 | 510.89 | 510.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 516.00 | 510.89 | 510.21 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 510.75 | 512.04 | 512.13 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 513.50 | 512.06 | 511.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 10:15:00 | 514.55 | 512.64 | 512.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 520.20 | 520.55 | 518.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 14:00:00 | 520.20 | 520.55 | 518.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 519.50 | 520.31 | 518.59 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 512.45 | 517.74 | 518.13 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 520.60 | 518.55 | 518.34 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 513.25 | 517.97 | 518.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 508.00 | 515.97 | 517.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 521.95 | 516.69 | 517.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 521.95 | 516.69 | 517.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 521.95 | 516.69 | 517.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 520.25 | 516.69 | 517.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 522.70 | 517.90 | 517.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 522.70 | 517.90 | 517.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 11:15:00 | 521.50 | 518.62 | 518.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 532.05 | 522.03 | 520.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 531.45 | 532.42 | 528.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 14:45:00 | 531.50 | 532.42 | 528.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 534.55 | 532.63 | 529.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 09:30:00 | 538.20 | 533.89 | 531.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:45:00 | 538.35 | 535.13 | 532.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 538.50 | 536.58 | 533.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 540.40 | 541.09 | 539.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 543.10 | 541.27 | 540.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:15:00 | 546.75 | 541.27 | 540.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 530.00 | 538.48 | 539.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 530.00 | 538.48 | 539.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 526.25 | 533.60 | 536.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 13:15:00 | 523.60 | 520.90 | 523.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 13:15:00 | 523.60 | 520.90 | 523.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 523.60 | 520.90 | 523.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 523.60 | 520.90 | 523.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 525.50 | 521.82 | 523.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 520.90 | 522.31 | 523.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 523.30 | 521.92 | 522.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 515.15 | 514.22 | 514.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 515.15 | 514.22 | 514.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 10:15:00 | 520.65 | 516.72 | 515.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 15:15:00 | 522.55 | 522.69 | 520.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 09:15:00 | 524.40 | 522.69 | 520.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 522.05 | 522.56 | 520.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 12:45:00 | 528.00 | 524.36 | 522.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 526.75 | 524.83 | 522.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 15:00:00 | 527.85 | 525.44 | 523.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 519.45 | 524.33 | 522.93 | SL hit (close<static) qty=1.00 sl=520.15 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 519.35 | 521.85 | 522.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 517.40 | 520.47 | 521.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 15:15:00 | 518.25 | 517.83 | 519.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 09:15:00 | 516.10 | 517.83 | 519.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 512.15 | 516.69 | 518.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:30:00 | 509.50 | 512.50 | 515.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 519.85 | 512.84 | 512.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 519.85 | 512.84 | 512.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 530.25 | 517.00 | 514.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 530.80 | 532.84 | 528.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 530.80 | 532.84 | 528.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 530.80 | 532.84 | 528.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:45:00 | 530.00 | 532.84 | 528.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 535.05 | 536.95 | 534.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 537.00 | 536.95 | 534.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 535.00 | 536.56 | 534.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 533.55 | 536.56 | 534.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 534.00 | 536.05 | 534.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 523.55 | 536.05 | 534.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 525.10 | 533.86 | 533.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 525.00 | 533.86 | 533.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 526.05 | 532.29 | 532.65 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 537.50 | 531.11 | 530.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 538.50 | 533.37 | 531.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 533.05 | 533.31 | 532.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 533.05 | 533.31 | 532.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 533.05 | 533.31 | 532.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 533.40 | 533.31 | 532.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 531.00 | 532.85 | 531.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 531.00 | 532.85 | 531.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 531.60 | 532.60 | 531.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 14:00:00 | 532.75 | 532.50 | 531.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 534.80 | 532.23 | 531.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 11:15:00 | 531.80 | 534.84 | 535.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 11:15:00 | 531.80 | 534.84 | 535.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 527.30 | 533.33 | 534.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 528.05 | 526.98 | 529.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 528.35 | 526.98 | 529.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 531.00 | 527.99 | 529.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 531.00 | 527.99 | 529.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 532.00 | 528.80 | 530.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 526.15 | 528.80 | 530.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 526.15 | 528.27 | 529.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 524.20 | 526.24 | 528.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 13:00:00 | 522.55 | 524.98 | 526.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:30:00 | 524.50 | 523.22 | 523.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 15:15:00 | 524.95 | 523.22 | 523.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 524.95 | 523.56 | 523.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 524.95 | 523.56 | 523.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 528.55 | 524.89 | 524.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 525.25 | 526.71 | 525.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 525.25 | 526.71 | 525.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 525.25 | 526.71 | 525.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 525.75 | 526.71 | 525.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 527.05 | 526.78 | 525.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:45:00 | 531.00 | 527.85 | 526.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 525.10 | 527.08 | 526.48 | SL hit (close<static) qty=1.00 sl=525.25 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 525.10 | 526.06 | 526.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 524.20 | 525.08 | 525.47 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 534.90 | 526.57 | 525.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 14:15:00 | 537.95 | 528.74 | 527.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 09:15:00 | 525.90 | 529.15 | 527.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 525.90 | 529.15 | 527.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 525.90 | 529.15 | 527.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 527.05 | 529.15 | 527.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 522.15 | 527.75 | 527.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:00:00 | 522.15 | 527.75 | 527.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 11:15:00 | 518.20 | 525.84 | 526.39 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 525.30 | 524.36 | 524.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 527.55 | 525.10 | 524.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 527.25 | 529.17 | 527.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 527.25 | 529.17 | 527.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 527.25 | 529.17 | 527.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 527.25 | 529.17 | 527.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 528.15 | 528.96 | 527.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 532.35 | 528.96 | 527.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 548.70 | 550.19 | 550.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 15:15:00 | 548.70 | 550.19 | 550.33 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 553.30 | 550.81 | 550.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 11:15:00 | 557.15 | 554.87 | 553.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 554.25 | 555.23 | 554.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 14:15:00 | 554.25 | 555.23 | 554.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 554.25 | 555.23 | 554.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 554.25 | 555.23 | 554.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 555.00 | 555.19 | 554.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 556.80 | 555.19 | 554.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:30:00 | 556.15 | 555.31 | 554.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:00:00 | 555.95 | 555.31 | 554.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 14:15:00 | 553.35 | 554.68 | 554.40 | SL hit (close<static) qty=1.00 sl=553.80 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 552.80 | 554.04 | 554.15 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 560.20 | 554.98 | 554.46 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 555.00 | 556.55 | 556.64 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 558.50 | 557.01 | 556.84 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 556.20 | 557.18 | 557.20 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 557.40 | 557.05 | 557.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 560.00 | 557.71 | 557.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 14:15:00 | 559.30 | 559.37 | 558.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 14:45:00 | 559.20 | 559.37 | 558.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 558.15 | 559.13 | 558.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 560.95 | 559.13 | 558.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 558.75 | 559.05 | 558.47 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 543.05 | 555.71 | 557.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 534.20 | 549.81 | 554.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 553.50 | 547.45 | 551.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 553.50 | 547.45 | 551.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 553.50 | 547.45 | 551.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 553.50 | 547.45 | 551.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 557.50 | 549.46 | 552.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 551.00 | 549.46 | 552.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 543.60 | 542.32 | 546.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:15:00 | 562.65 | 542.32 | 546.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 559.40 | 545.74 | 548.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 542.20 | 547.31 | 548.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 539.15 | 531.75 | 531.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 539.15 | 531.75 | 531.32 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 529.60 | 534.17 | 534.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 527.35 | 531.95 | 533.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 534.45 | 531.50 | 532.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 534.45 | 531.50 | 532.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 534.45 | 531.50 | 532.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 534.45 | 531.50 | 532.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 533.30 | 531.86 | 532.85 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 539.45 | 534.19 | 533.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 542.00 | 535.75 | 534.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 523.65 | 534.14 | 534.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 523.65 | 534.14 | 534.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 523.65 | 534.14 | 534.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 523.65 | 534.14 | 534.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 517.55 | 530.82 | 532.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 514.65 | 527.59 | 530.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 488.00 | 486.24 | 492.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 488.00 | 486.24 | 492.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 515.90 | 492.85 | 494.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 515.90 | 492.85 | 494.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 526.95 | 499.67 | 497.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 546.30 | 526.53 | 521.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 661.50 | 662.26 | 637.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:00:00 | 661.50 | 662.26 | 637.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 659.00 | 665.90 | 655.44 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 639.70 | 653.51 | 653.60 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 660.00 | 654.32 | 653.79 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 646.90 | 653.10 | 653.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 642.05 | 649.27 | 651.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 655.45 | 650.47 | 651.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 655.45 | 650.47 | 651.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 655.45 | 650.47 | 651.65 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 667.40 | 655.30 | 653.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 710.00 | 668.04 | 660.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 716.25 | 719.57 | 705.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 716.25 | 719.57 | 705.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 722.90 | 724.46 | 719.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 718.70 | 724.46 | 719.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 733.45 | 726.47 | 721.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 745.00 | 728.47 | 724.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:00:00 | 743.20 | 734.06 | 728.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 741.25 | 737.91 | 733.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 749.75 | 754.07 | 750.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 752.80 | 753.82 | 750.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:45:00 | 754.00 | 753.17 | 750.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 763.00 | 751.77 | 750.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 763.40 | 753.49 | 751.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 09:15:00 | 819.50 | 769.01 | 760.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 756.10 | 779.75 | 782.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 751.25 | 766.97 | 772.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 784.80 | 770.53 | 773.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 784.80 | 770.53 | 773.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 784.80 | 770.53 | 773.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 784.80 | 770.53 | 773.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 775.00 | 771.43 | 773.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 779.70 | 771.43 | 773.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 770.30 | 772.09 | 773.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 771.95 | 772.09 | 773.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 779.90 | 773.65 | 774.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 779.90 | 773.65 | 774.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 783.00 | 775.52 | 775.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 802.55 | 782.15 | 778.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 803.50 | 804.42 | 797.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 803.40 | 804.42 | 797.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 795.40 | 802.62 | 797.07 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 791.70 | 795.67 | 795.95 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 807.50 | 798.34 | 797.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 817.45 | 803.69 | 800.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 847.90 | 849.15 | 833.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 847.90 | 849.15 | 833.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 903.60 | 911.38 | 901.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:45:00 | 915.20 | 909.81 | 902.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 925.80 | 911.94 | 904.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 916.00 | 916.48 | 914.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 913.80 | 916.74 | 915.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 911.10 | 915.61 | 914.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 911.25 | 915.61 | 914.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 904.50 | 913.39 | 913.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 904.50 | 913.39 | 913.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 895.00 | 909.71 | 912.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 885.00 | 870.20 | 878.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 885.00 | 870.20 | 878.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 885.00 | 870.20 | 878.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 884.00 | 870.20 | 878.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 888.25 | 873.81 | 879.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 891.65 | 873.81 | 879.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 895.00 | 883.59 | 883.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 902.00 | 887.27 | 884.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 889.15 | 889.99 | 886.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:45:00 | 889.00 | 889.99 | 886.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 887.25 | 889.44 | 886.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 886.20 | 889.44 | 886.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 882.55 | 888.06 | 886.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 882.55 | 888.06 | 886.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 876.55 | 885.76 | 885.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 878.85 | 885.76 | 885.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 872.05 | 883.02 | 884.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 866.80 | 876.63 | 880.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 15:15:00 | 873.50 | 872.76 | 876.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 869.15 | 872.76 | 876.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 867.00 | 871.61 | 875.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 852.70 | 865.56 | 868.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 810.07 | 821.85 | 834.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 824.30 | 822.34 | 833.78 | SL hit (close>ema200) qty=0.50 sl=822.34 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 780.90 | 765.93 | 765.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 15:15:00 | 788.00 | 776.44 | 771.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 776.55 | 784.13 | 778.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 776.55 | 784.13 | 778.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 776.55 | 784.13 | 778.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 776.55 | 784.13 | 778.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 779.00 | 783.11 | 778.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 806.50 | 783.11 | 778.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:45:00 | 780.00 | 791.11 | 789.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 775.30 | 786.79 | 788.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 775.30 | 786.79 | 788.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 757.95 | 777.69 | 783.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 10:15:00 | 746.20 | 743.15 | 752.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:45:00 | 746.20 | 743.15 | 752.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 737.00 | 734.24 | 741.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 740.75 | 734.24 | 741.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 752.05 | 734.59 | 738.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 753.00 | 734.59 | 738.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 755.60 | 738.79 | 740.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 756.15 | 738.79 | 740.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 762.45 | 743.52 | 742.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 768.35 | 752.95 | 747.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 761.00 | 764.91 | 757.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 761.00 | 764.91 | 757.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 761.00 | 764.91 | 757.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:45:00 | 758.65 | 764.91 | 757.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 758.50 | 763.63 | 757.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 771.85 | 763.63 | 757.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 773.40 | 778.51 | 779.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 773.40 | 778.51 | 779.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 770.80 | 776.97 | 778.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 769.75 | 763.06 | 768.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 769.75 | 763.06 | 768.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 769.75 | 763.06 | 768.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 769.75 | 763.06 | 768.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 764.40 | 763.33 | 767.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 762.75 | 766.29 | 767.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 14:15:00 | 724.61 | 746.96 | 756.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 723.10 | 713.59 | 722.19 | SL hit (close>ema200) qty=0.50 sl=713.59 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 719.55 | 714.23 | 713.59 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 709.00 | 716.19 | 716.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 701.70 | 711.98 | 714.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 722.75 | 712.74 | 714.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 12:15:00 | 722.75 | 712.74 | 714.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 722.75 | 712.74 | 714.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 722.95 | 712.74 | 714.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 723.30 | 714.85 | 714.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 723.30 | 714.85 | 714.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 722.85 | 716.45 | 715.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 724.70 | 718.97 | 717.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 788.55 | 790.85 | 777.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 12:45:00 | 788.75 | 790.85 | 777.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 788.70 | 789.26 | 780.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 782.25 | 789.26 | 780.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 782.85 | 786.94 | 781.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 780.15 | 786.94 | 781.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 781.70 | 785.89 | 781.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 785.20 | 785.89 | 781.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 776.90 | 784.09 | 780.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 776.90 | 784.09 | 780.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 782.10 | 783.70 | 780.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 778.10 | 783.70 | 780.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 783.25 | 783.61 | 781.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 789.65 | 783.61 | 781.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 788.95 | 784.68 | 781.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:45:00 | 788.85 | 786.54 | 783.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 802.00 | 813.00 | 813.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 802.00 | 813.00 | 813.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 798.85 | 810.17 | 812.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 792.05 | 791.84 | 800.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 792.05 | 791.84 | 800.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 803.25 | 794.12 | 801.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 804.00 | 794.12 | 801.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 802.65 | 795.83 | 801.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 802.65 | 795.83 | 801.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 800.40 | 796.74 | 801.15 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 825.20 | 806.90 | 804.99 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 801.45 | 811.05 | 811.85 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 15:15:00 | 820.00 | 812.15 | 811.50 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 803.25 | 810.10 | 810.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 800.00 | 808.08 | 809.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 794.05 | 791.60 | 798.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 794.05 | 791.60 | 798.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 794.05 | 791.60 | 798.04 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 817.10 | 801.86 | 801.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 850.85 | 822.84 | 815.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 839.60 | 842.38 | 831.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:45:00 | 842.50 | 842.38 | 831.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 835.55 | 841.02 | 832.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 837.10 | 841.02 | 832.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 832.40 | 839.29 | 832.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 832.40 | 839.29 | 832.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 809.85 | 833.40 | 830.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 807.05 | 833.40 | 830.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 820.85 | 830.89 | 829.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 826.20 | 830.89 | 829.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 814.55 | 826.02 | 827.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 814.55 | 826.02 | 827.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 810.30 | 822.87 | 825.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 810.10 | 807.77 | 813.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 15:15:00 | 810.10 | 807.77 | 813.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 810.10 | 807.77 | 813.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 819.95 | 807.77 | 813.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 817.70 | 809.76 | 813.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 824.00 | 809.76 | 813.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 817.60 | 811.33 | 813.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 819.20 | 811.33 | 813.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 818.65 | 815.69 | 815.53 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 810.45 | 815.04 | 815.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 803.95 | 812.83 | 814.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 11:15:00 | 774.20 | 771.68 | 784.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:00:00 | 774.20 | 771.68 | 784.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 782.15 | 774.46 | 783.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 782.15 | 774.46 | 783.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 789.35 | 777.44 | 783.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 789.35 | 777.44 | 783.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 785.00 | 778.95 | 784.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 788.30 | 778.95 | 784.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 783.55 | 779.87 | 784.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 778.30 | 779.87 | 784.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 793.00 | 780.71 | 779.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 793.00 | 780.71 | 779.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 797.80 | 790.00 | 784.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 798.90 | 804.30 | 798.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 798.90 | 804.30 | 798.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 798.90 | 804.30 | 798.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 798.90 | 804.30 | 798.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 797.30 | 802.90 | 798.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 793.60 | 802.90 | 798.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 792.85 | 800.89 | 797.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 792.85 | 800.89 | 797.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 811.05 | 802.92 | 798.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 817.40 | 802.92 | 798.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 816.80 | 833.90 | 835.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 816.80 | 833.90 | 835.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 808.20 | 828.76 | 833.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 840.10 | 825.87 | 830.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 840.10 | 825.87 | 830.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 840.10 | 825.87 | 830.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 840.10 | 825.87 | 830.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 836.25 | 827.95 | 830.91 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 850.30 | 834.87 | 833.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 858.50 | 839.59 | 835.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 857.30 | 858.16 | 850.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 857.30 | 858.16 | 850.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 858.10 | 858.76 | 852.98 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 830.05 | 850.04 | 850.59 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 856.45 | 850.34 | 850.31 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 832.85 | 847.29 | 848.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 10:15:00 | 829.20 | 843.67 | 847.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 826.60 | 818.44 | 826.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 826.60 | 818.44 | 826.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 826.60 | 818.44 | 826.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 826.60 | 818.44 | 826.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 826.80 | 820.11 | 826.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 826.80 | 820.11 | 826.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 827.10 | 821.51 | 826.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:00:00 | 823.50 | 821.91 | 826.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 824.05 | 822.81 | 824.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 824.90 | 822.81 | 824.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:45:00 | 824.60 | 823.72 | 824.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 820.50 | 823.08 | 824.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 817.40 | 823.08 | 824.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 830.00 | 824.16 | 824.40 | SL hit (close>static) qty=1.00 sl=828.65 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 828.30 | 824.99 | 824.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 849.40 | 830.39 | 827.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 831.95 | 833.62 | 829.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 12:00:00 | 831.95 | 833.62 | 829.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 828.35 | 832.56 | 829.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 828.35 | 832.56 | 829.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 841.70 | 834.39 | 830.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 827.50 | 834.39 | 830.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 805.00 | 829.02 | 829.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 803.20 | 814.97 | 821.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 810.20 | 810.06 | 817.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 810.20 | 810.06 | 817.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 810.20 | 810.06 | 817.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 812.75 | 810.06 | 817.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 819.65 | 811.62 | 816.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 819.65 | 811.62 | 816.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 820.50 | 813.40 | 817.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 816.05 | 813.68 | 816.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 814.80 | 813.68 | 816.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 815.70 | 814.27 | 816.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 775.25 | 793.26 | 802.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 774.06 | 791.01 | 800.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 774.91 | 791.01 | 800.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 777.55 | 777.05 | 786.50 | SL hit (close>ema200) qty=0.50 sl=777.05 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 790.10 | 784.30 | 784.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 803.00 | 791.44 | 787.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 844.10 | 855.81 | 842.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:45:00 | 844.10 | 855.81 | 842.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 848.95 | 852.29 | 846.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 836.00 | 852.29 | 846.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 835.05 | 848.84 | 845.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 835.05 | 848.84 | 845.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 830.30 | 845.13 | 843.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 830.45 | 845.13 | 843.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 832.55 | 842.62 | 842.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 829.65 | 837.01 | 839.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 821.80 | 818.48 | 825.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:30:00 | 821.80 | 818.48 | 825.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 824.80 | 819.75 | 825.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 824.80 | 819.75 | 825.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 821.30 | 820.06 | 824.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 823.00 | 820.06 | 824.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 804.85 | 817.20 | 822.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 803.45 | 813.12 | 819.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:30:00 | 801.50 | 800.79 | 809.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 803.50 | 802.51 | 809.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 826.40 | 813.57 | 812.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 826.40 | 813.57 | 812.70 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 806.40 | 813.01 | 813.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 802.65 | 810.94 | 812.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 809.20 | 809.16 | 811.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 809.20 | 809.16 | 811.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 809.20 | 809.16 | 811.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 809.20 | 809.16 | 811.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 814.70 | 810.27 | 811.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 816.40 | 810.27 | 811.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 823.80 | 812.97 | 812.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 830.45 | 822.44 | 817.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 827.80 | 830.17 | 825.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 12:15:00 | 827.80 | 830.17 | 825.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 827.80 | 830.17 | 825.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 827.80 | 830.17 | 825.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 825.50 | 829.23 | 825.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:45:00 | 826.45 | 829.23 | 825.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 821.30 | 827.65 | 825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 820.40 | 827.65 | 825.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 828.00 | 827.72 | 825.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 816.55 | 827.72 | 825.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 815.25 | 825.22 | 824.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 818.10 | 825.22 | 824.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 817.55 | 823.69 | 824.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 813.60 | 821.67 | 823.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 793.65 | 792.82 | 799.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 794.00 | 792.82 | 799.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 793.60 | 792.97 | 799.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 791.00 | 792.97 | 799.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 813.00 | 800.32 | 800.33 | SL hit (close>static) qty=1.00 sl=800.95 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 817.60 | 803.78 | 801.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 821.60 | 807.34 | 803.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 816.55 | 817.04 | 811.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 816.55 | 817.04 | 811.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 819.20 | 820.31 | 816.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 821.30 | 820.31 | 816.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 821.30 | 820.51 | 816.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 812.50 | 817.84 | 816.26 | SL hit (close<static) qty=1.00 sl=816.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 806.70 | 813.64 | 814.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 804.00 | 811.72 | 813.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 776.20 | 774.94 | 784.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 776.20 | 774.94 | 784.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 788.75 | 778.07 | 782.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 788.75 | 778.07 | 782.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 785.00 | 779.46 | 782.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 787.00 | 779.46 | 782.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 782.95 | 780.16 | 782.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 780.00 | 780.07 | 781.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 779.60 | 775.96 | 777.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 741.00 | 750.77 | 755.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 740.62 | 750.77 | 755.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 11:15:00 | 702.00 | 724.66 | 738.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 636.75 | 629.39 | 629.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 644.40 | 633.09 | 630.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 637.00 | 639.16 | 635.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 637.00 | 639.16 | 635.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 637.00 | 639.16 | 635.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 641.05 | 639.16 | 635.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 643.35 | 640.00 | 636.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:15:00 | 648.80 | 640.00 | 636.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 624.60 | 634.98 | 636.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 624.60 | 634.98 | 636.11 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 678.00 | 640.35 | 637.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 680.75 | 648.43 | 641.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 681.55 | 687.22 | 673.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 681.35 | 687.22 | 673.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 674.05 | 684.58 | 673.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 674.05 | 684.58 | 673.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 672.70 | 682.21 | 673.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:15:00 | 681.20 | 680.62 | 673.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 682.50 | 679.15 | 674.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 647.70 | 673.40 | 672.50 | SL hit (close<static) qty=1.00 sl=670.20 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 630.30 | 664.78 | 668.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 620.80 | 649.96 | 660.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 638.70 | 638.67 | 650.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 638.45 | 638.67 | 650.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 662.20 | 644.21 | 647.98 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 669.30 | 652.32 | 651.20 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 636.40 | 649.33 | 651.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 617.95 | 635.69 | 642.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 603.05 | 597.19 | 606.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 603.05 | 597.19 | 606.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 603.05 | 597.19 | 606.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 603.20 | 597.19 | 606.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 602.00 | 598.60 | 604.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:30:00 | 606.70 | 598.60 | 604.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 604.00 | 599.68 | 604.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 605.35 | 599.68 | 604.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 602.80 | 600.31 | 604.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:00:00 | 601.30 | 601.05 | 603.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:00:00 | 601.30 | 601.10 | 603.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 600.80 | 601.80 | 603.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 605.75 | 602.80 | 603.80 | SL hit (close>static) qty=1.00 sl=605.10 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 605.50 | 596.83 | 596.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 608.10 | 599.08 | 597.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 596.25 | 600.85 | 599.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 10:15:00 | 596.25 | 600.85 | 599.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 596.25 | 600.85 | 599.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 596.00 | 600.85 | 599.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 592.45 | 599.17 | 598.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 592.45 | 599.17 | 598.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 592.35 | 597.80 | 598.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 15:15:00 | 591.00 | 594.99 | 596.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 598.50 | 594.78 | 595.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 598.50 | 594.78 | 595.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 598.50 | 594.78 | 595.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 602.80 | 594.78 | 595.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 591.55 | 594.13 | 595.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:15:00 | 589.50 | 594.13 | 595.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 588.50 | 592.35 | 594.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 589.75 | 591.55 | 593.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 604.00 | 594.04 | 594.77 | SL hit (close>static) qty=1.00 sl=599.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 604.25 | 596.09 | 595.63 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 591.75 | 595.26 | 595.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 582.35 | 591.34 | 593.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 569.20 | 569.04 | 577.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 569.20 | 569.04 | 577.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 563.35 | 557.29 | 561.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 563.35 | 557.29 | 561.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 564.00 | 558.63 | 561.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 562.25 | 558.63 | 561.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 573.05 | 561.52 | 562.93 | SL hit (close>static) qty=1.00 sl=570.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 570.00 | 564.80 | 564.28 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 547.85 | 562.49 | 563.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 541.95 | 553.16 | 558.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 552.95 | 550.57 | 555.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 552.95 | 550.57 | 555.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 552.95 | 550.57 | 555.35 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 564.00 | 557.74 | 557.13 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 552.00 | 557.45 | 557.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 541.60 | 554.28 | 556.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 555.60 | 554.10 | 555.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 555.60 | 554.10 | 555.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 555.60 | 554.10 | 555.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 556.05 | 554.10 | 555.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 556.70 | 554.62 | 555.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 559.10 | 554.62 | 555.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 554.50 | 554.60 | 555.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 557.60 | 554.60 | 555.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 546.10 | 552.90 | 554.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 549.95 | 552.90 | 554.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 547.65 | 551.16 | 553.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:30:00 | 559.70 | 551.16 | 553.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 538.00 | 546.65 | 550.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 547.30 | 546.65 | 550.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 534.50 | 530.80 | 537.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 535.00 | 530.80 | 537.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 536.50 | 531.94 | 537.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 533.40 | 531.94 | 537.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 548.50 | 534.57 | 535.74 | SL hit (close>static) qty=1.00 sl=540.95 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 549.00 | 537.46 | 536.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 552.00 | 540.37 | 538.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 543.30 | 544.80 | 541.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 543.30 | 544.80 | 541.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 543.30 | 544.80 | 541.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 558.75 | 544.18 | 542.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 548.45 | 544.43 | 542.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 548.20 | 544.86 | 543.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 536.00 | 542.97 | 542.93 | SL hit (close<static) qty=1.00 sl=538.10 alert=retest2 |

### Cycle 136 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 518.55 | 538.09 | 540.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 511.15 | 532.70 | 538.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 524.15 | 518.33 | 526.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 524.15 | 518.33 | 526.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 524.15 | 518.33 | 526.75 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 537.00 | 529.80 | 529.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 546.45 | 533.13 | 530.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 540.25 | 549.37 | 542.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 540.25 | 549.37 | 542.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 540.25 | 549.37 | 542.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 538.40 | 549.37 | 542.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 535.50 | 546.59 | 541.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 535.50 | 546.59 | 541.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 532.55 | 543.79 | 540.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 532.70 | 543.79 | 540.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 529.15 | 538.83 | 539.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 517.50 | 534.56 | 537.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 525.85 | 521.97 | 527.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 525.85 | 521.97 | 527.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 525.85 | 521.97 | 527.80 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 538.00 | 531.40 | 530.56 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 513.30 | 527.78 | 528.99 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 537.65 | 529.82 | 529.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 541.50 | 532.15 | 530.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 597.65 | 598.93 | 585.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 598.60 | 598.93 | 585.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 618.00 | 621.62 | 608.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 621.55 | 621.37 | 609.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 621.60 | 621.37 | 609.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 623.00 | 622.29 | 612.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 631.95 | 621.33 | 614.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 649.85 | 648.76 | 643.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 641.25 | 648.76 | 643.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 640.45 | 646.95 | 643.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 640.45 | 646.95 | 643.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 639.35 | 645.43 | 643.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 639.35 | 645.43 | 643.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 644.85 | 647.80 | 645.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 644.85 | 647.80 | 645.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 643.50 | 646.94 | 645.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 647.00 | 646.94 | 645.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:00:00 | 647.05 | 646.96 | 645.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 648.45 | 647.11 | 646.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 13:15:00 | 683.71 | 659.66 | 652.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 645.85 | 657.50 | 658.35 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 663.10 | 656.71 | 656.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 668.70 | 659.11 | 657.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 680.10 | 680.68 | 673.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 680.10 | 680.68 | 673.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 353.05 | 2024-05-14 09:15:00 | 373.00 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest2 | 2024-05-16 09:15:00 | 384.90 | 2024-05-29 15:15:00 | 393.00 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2024-06-11 09:15:00 | 443.80 | 2024-06-18 12:15:00 | 488.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 14:45:00 | 520.40 | 2024-07-02 09:15:00 | 572.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-19 11:45:00 | 500.80 | 2024-07-22 11:15:00 | 500.25 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-08-01 09:15:00 | 530.00 | 2024-08-01 11:15:00 | 513.85 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-08-14 12:15:00 | 563.25 | 2024-08-16 14:15:00 | 542.05 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2024-08-14 13:15:00 | 562.45 | 2024-08-16 14:15:00 | 542.05 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest1 | 2024-08-22 09:15:00 | 544.75 | 2024-08-22 09:15:00 | 556.85 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-08-23 10:30:00 | 546.50 | 2024-08-26 11:15:00 | 555.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-09-02 12:15:00 | 553.30 | 2024-09-06 09:15:00 | 525.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 552.80 | 2024-09-06 09:15:00 | 525.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 10:00:00 | 551.25 | 2024-09-06 09:15:00 | 523.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 12:15:00 | 553.30 | 2024-09-09 09:15:00 | 497.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 552.80 | 2024-09-09 09:15:00 | 497.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-03 10:00:00 | 551.25 | 2024-09-09 09:15:00 | 496.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 15:15:00 | 542.00 | 2024-10-03 09:15:00 | 620.00 | STOP_HIT | 1.00 | -14.39% |
| SELL | retest2 | 2024-10-14 11:00:00 | 577.85 | 2024-10-16 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2024-10-14 12:00:00 | 576.80 | 2024-10-16 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.58% |
| SELL | retest2 | 2024-10-15 10:15:00 | 578.90 | 2024-10-16 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2024-10-15 12:30:00 | 578.00 | 2024-10-16 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2024-10-18 15:15:00 | 582.00 | 2024-10-22 10:15:00 | 552.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:15:00 | 582.00 | 2024-10-23 09:15:00 | 556.90 | STOP_HIT | 0.50 | 4.31% |
| BUY | retest2 | 2024-11-06 11:15:00 | 568.10 | 2024-11-07 11:15:00 | 550.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-11-06 12:15:00 | 564.25 | 2024-11-07 11:15:00 | 550.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-11-06 13:15:00 | 564.45 | 2024-11-07 11:15:00 | 550.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-11-06 14:00:00 | 564.45 | 2024-11-07 11:15:00 | 550.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-11-08 14:00:00 | 545.70 | 2024-11-14 09:15:00 | 518.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 15:00:00 | 542.20 | 2024-11-18 09:15:00 | 515.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 545.70 | 2024-11-18 13:15:00 | 518.25 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2024-11-08 15:00:00 | 542.20 | 2024-11-18 13:15:00 | 518.25 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2024-11-29 12:00:00 | 514.50 | 2024-12-04 10:15:00 | 512.15 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-11-29 12:30:00 | 517.00 | 2024-12-04 10:15:00 | 512.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-09 14:15:00 | 505.60 | 2024-12-10 15:15:00 | 516.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-12-10 13:00:00 | 506.20 | 2024-12-10 15:15:00 | 516.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-12-31 09:30:00 | 538.20 | 2025-01-06 09:15:00 | 530.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-31 12:45:00 | 538.35 | 2025-01-06 09:15:00 | 530.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-31 15:15:00 | 538.50 | 2025-01-06 09:15:00 | 530.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-01-02 13:30:00 | 540.40 | 2025-01-06 09:15:00 | 530.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-01-03 10:15:00 | 546.75 | 2025-01-06 09:15:00 | 530.00 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-01-10 09:15:00 | 520.90 | 2025-01-16 10:15:00 | 515.15 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2025-01-10 14:00:00 | 523.30 | 2025-01-16 10:15:00 | 515.15 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-01-21 12:45:00 | 528.00 | 2025-01-22 09:15:00 | 519.45 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-01-21 14:00:00 | 526.75 | 2025-01-22 09:15:00 | 519.45 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-01-21 15:00:00 | 527.85 | 2025-01-22 09:15:00 | 519.45 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-01-27 09:30:00 | 509.50 | 2025-01-28 13:15:00 | 519.85 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-02-05 14:00:00 | 532.75 | 2025-02-11 11:15:00 | 531.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-02-06 09:15:00 | 534.80 | 2025-02-11 11:15:00 | 531.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-02-13 15:15:00 | 524.20 | 2025-02-19 15:15:00 | 524.95 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-02-14 13:00:00 | 522.55 | 2025-02-19 15:15:00 | 524.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-02-19 14:30:00 | 524.50 | 2025-02-19 15:15:00 | 524.95 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-02-19 15:15:00 | 524.95 | 2025-02-19 15:15:00 | 524.95 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-02-21 12:45:00 | 531.00 | 2025-02-24 10:15:00 | 525.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-03-07 13:15:00 | 532.35 | 2025-03-18 15:15:00 | 548.70 | STOP_HIT | 1.00 | 3.07% |
| BUY | retest2 | 2025-03-24 09:15:00 | 556.80 | 2025-03-24 14:15:00 | 553.35 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-03-24 10:30:00 | 556.15 | 2025-03-24 14:15:00 | 553.35 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-03-24 11:00:00 | 555.95 | 2025-03-24 14:15:00 | 553.35 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-04-09 10:00:00 | 542.20 | 2025-04-21 09:15:00 | 539.15 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-06-02 15:15:00 | 745.00 | 2025-06-10 09:15:00 | 819.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-03 10:00:00 | 743.20 | 2025-06-10 09:15:00 | 817.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 741.25 | 2025-06-10 09:15:00 | 815.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 10:15:00 | 749.75 | 2025-06-10 09:15:00 | 824.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 13:45:00 | 754.00 | 2025-06-10 09:15:00 | 829.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 09:30:00 | 763.00 | 2025-06-12 13:15:00 | 756.10 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-09 10:45:00 | 763.40 | 2025-06-12 13:15:00 | 756.10 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-02 09:45:00 | 915.20 | 2025-07-04 14:15:00 | 904.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-02 11:30:00 | 925.80 | 2025-07-04 14:15:00 | 904.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-04 09:45:00 | 916.00 | 2025-07-04 14:15:00 | 904.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-04 13:00:00 | 913.80 | 2025-07-04 14:15:00 | 904.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-16 09:15:00 | 852.70 | 2025-07-18 10:15:00 | 810.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 09:15:00 | 852.70 | 2025-07-18 11:15:00 | 824.30 | STOP_HIT | 0.50 | 3.33% |
| BUY | retest2 | 2025-08-01 09:15:00 | 806.50 | 2025-08-04 13:15:00 | 775.30 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-08-04 11:45:00 | 780.00 | 2025-08-04 13:15:00 | 775.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-13 09:15:00 | 771.85 | 2025-08-20 13:15:00 | 773.40 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-25 15:15:00 | 762.75 | 2025-08-26 14:15:00 | 724.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 15:15:00 | 762.75 | 2025-09-01 09:15:00 | 723.10 | STOP_HIT | 0.50 | 5.20% |
| BUY | retest2 | 2025-09-15 09:15:00 | 789.65 | 2025-09-19 12:15:00 | 802.00 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-09-15 10:00:00 | 788.95 | 2025-09-19 12:15:00 | 802.00 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-09-15 12:45:00 | 788.85 | 2025-09-19 12:15:00 | 802.00 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2025-10-07 14:15:00 | 826.20 | 2025-10-08 09:15:00 | 814.55 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-16 10:15:00 | 778.30 | 2025-10-20 13:15:00 | 793.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-10-27 11:15:00 | 817.40 | 2025-10-31 12:15:00 | 816.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-12 13:00:00 | 823.50 | 2025-11-14 11:15:00 | 830.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-13 12:30:00 | 824.05 | 2025-11-14 12:15:00 | 828.30 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-13 13:00:00 | 824.90 | 2025-11-14 12:15:00 | 828.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-13 13:45:00 | 824.60 | 2025-11-14 12:15:00 | 828.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-13 15:15:00 | 817.40 | 2025-11-14 12:15:00 | 828.30 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-11-19 13:30:00 | 816.05 | 2025-11-21 14:15:00 | 775.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 14:00:00 | 814.80 | 2025-11-21 15:15:00 | 774.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 815.70 | 2025-11-21 15:15:00 | 774.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 13:30:00 | 816.05 | 2025-11-25 09:15:00 | 777.55 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-11-19 14:00:00 | 814.80 | 2025-11-25 09:15:00 | 777.55 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2025-11-19 15:15:00 | 815.70 | 2025-11-25 09:15:00 | 777.55 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-12-08 11:45:00 | 803.45 | 2025-12-09 15:15:00 | 826.40 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-12-09 10:30:00 | 801.50 | 2025-12-09 15:15:00 | 826.40 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-12-09 11:30:00 | 803.50 | 2025-12-09 15:15:00 | 826.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-12-19 11:15:00 | 791.00 | 2025-12-22 09:15:00 | 813.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-12-24 11:15:00 | 821.30 | 2025-12-24 14:15:00 | 812.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-24 12:00:00 | 821.30 | 2025-12-24 14:15:00 | 812.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-01 11:30:00 | 780.00 | 2026-01-08 11:15:00 | 741.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 14:15:00 | 779.60 | 2026-01-08 11:15:00 | 740.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 11:30:00 | 780.00 | 2026-01-09 11:15:00 | 702.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-02 14:15:00 | 779.60 | 2026-01-09 11:15:00 | 701.64 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 10:15:00 | 648.80 | 2026-02-02 10:15:00 | 624.60 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2026-02-05 13:15:00 | 681.20 | 2026-02-06 09:15:00 | 647.70 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2026-02-05 15:15:00 | 682.50 | 2026-02-06 09:15:00 | 647.70 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-02-18 10:00:00 | 601.30 | 2026-02-18 14:15:00 | 605.75 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-18 11:00:00 | 601.30 | 2026-02-18 14:15:00 | 605.75 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-18 11:45:00 | 600.80 | 2026-02-18 14:15:00 | 605.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-19 09:30:00 | 599.65 | 2026-02-23 11:15:00 | 605.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-25 13:15:00 | 589.50 | 2026-02-26 09:15:00 | 604.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-25 13:45:00 | 588.50 | 2026-02-26 09:15:00 | 604.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-02-26 09:15:00 | 589.75 | 2026-02-26 09:15:00 | 604.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-03-06 09:15:00 | 562.25 | 2026-03-06 09:15:00 | 573.05 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-03-17 11:15:00 | 533.40 | 2026-03-18 10:15:00 | 548.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-20 09:15:00 | 558.75 | 2026-03-20 15:15:00 | 536.00 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2026-03-20 10:15:00 | 548.45 | 2026-03-20 15:15:00 | 536.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-03-20 11:15:00 | 548.20 | 2026-03-20 15:15:00 | 536.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-04-13 10:30:00 | 621.55 | 2026-04-22 13:15:00 | 683.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:15:00 | 621.60 | 2026-04-22 13:15:00 | 683.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:30:00 | 623.00 | 2026-04-22 13:15:00 | 685.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 631.95 | 2026-04-24 11:15:00 | 645.85 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2026-04-22 09:15:00 | 647.00 | 2026-04-24 11:15:00 | 645.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-04-22 10:00:00 | 647.05 | 2026-04-24 11:15:00 | 645.85 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-04-22 10:45:00 | 648.45 | 2026-04-24 11:15:00 | 645.85 | STOP_HIT | 1.00 | -0.40% |
