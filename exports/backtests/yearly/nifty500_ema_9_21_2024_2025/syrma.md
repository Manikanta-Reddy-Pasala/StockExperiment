# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1100.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 127 |
| ALERT1 | 88 |
| ALERT2 | 88 |
| ALERT2_SKIP | 48 |
| ALERT3 | 208 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 114 |
| PARTIAL | 30 |
| TARGET_HIT | 13 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 76
- **Target hits / Stop hits / Partials:** 11 / 108 / 28
- **Avg / median % per leg:** 1.12% / -0.24%
- **Sum % (uncompounded):** 164.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 16 | 29.1% | 8 | 46 | 1 | 0.23% | 12.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.39% | -2.3% |
| BUY @ 3rd Alert (retest2) | 49 | 14 | 28.6% | 8 | 41 | 0 | 0.31% | 15.0% |
| SELL (all) | 92 | 55 | 59.8% | 3 | 62 | 27 | 1.65% | 151.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 92 | 55 | 59.8% | 3 | 62 | 27 | 1.65% | 151.9% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.39% | -2.3% |
| retest2 (combined) | 141 | 69 | 48.9% | 11 | 103 | 27 | 1.18% | 166.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 12:15:00 | 412.00 | 404.03 | 403.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 413.05 | 407.11 | 404.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 407.95 | 407.97 | 405.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 407.95 | 407.97 | 405.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 471.00 | 474.17 | 468.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 470.85 | 474.17 | 468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 467.30 | 472.02 | 468.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 467.30 | 472.02 | 468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 468.20 | 471.26 | 468.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 474.15 | 471.26 | 468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 472.70 | 471.54 | 468.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 15:00:00 | 475.90 | 473.56 | 470.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 11:00:00 | 476.40 | 474.61 | 472.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 479.20 | 475.06 | 473.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 478.00 | 490.65 | 485.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 462.35 | 484.99 | 483.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 462.35 | 484.99 | 483.60 | SL hit (close<static) qty=1.00 sl=462.60 alert=retest2 |

### Cycle 2 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 445.05 | 477.00 | 480.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 12:15:00 | 424.60 | 466.52 | 475.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 440.00 | 439.16 | 451.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 448.25 | 439.16 | 451.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 452.05 | 441.74 | 451.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:15:00 | 454.20 | 441.74 | 451.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 453.85 | 444.16 | 451.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 453.85 | 444.16 | 451.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 453.00 | 445.93 | 451.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 452.50 | 447.24 | 451.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:30:00 | 452.60 | 449.35 | 451.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 473.30 | 454.89 | 454.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 473.30 | 454.89 | 454.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 481.00 | 471.90 | 470.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 11:15:00 | 471.00 | 472.11 | 470.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 12:00:00 | 471.00 | 472.11 | 470.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 471.80 | 472.04 | 471.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:45:00 | 474.30 | 472.04 | 471.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 470.25 | 471.69 | 470.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:30:00 | 472.00 | 471.69 | 470.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 470.85 | 471.52 | 470.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:30:00 | 469.95 | 471.52 | 470.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 471.50 | 471.51 | 470.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 475.55 | 471.64 | 471.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 474.30 | 472.01 | 471.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-21 11:15:00 | 523.11 | 491.26 | 482.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 494.45 | 495.78 | 495.90 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 496.80 | 495.98 | 495.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 500.85 | 496.97 | 496.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 504.50 | 505.44 | 501.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 14:00:00 | 504.50 | 505.44 | 501.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 504.70 | 505.30 | 502.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 503.25 | 505.30 | 502.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 503.60 | 504.83 | 502.71 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 496.80 | 501.22 | 501.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 09:15:00 | 495.95 | 499.49 | 500.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 491.65 | 488.72 | 491.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 491.65 | 488.72 | 491.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 491.65 | 488.72 | 491.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:15:00 | 495.50 | 488.72 | 491.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 497.40 | 490.46 | 492.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:15:00 | 499.60 | 490.46 | 492.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 497.20 | 491.80 | 492.76 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 498.85 | 494.13 | 493.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 502.30 | 496.15 | 494.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 502.80 | 503.27 | 499.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 09:30:00 | 502.75 | 503.27 | 499.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 501.50 | 503.64 | 501.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 501.50 | 503.64 | 501.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 502.00 | 503.31 | 501.30 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 494.25 | 499.32 | 499.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 490.15 | 495.42 | 497.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 491.40 | 490.10 | 492.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 491.40 | 490.10 | 492.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 491.40 | 490.10 | 492.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 491.40 | 490.10 | 492.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 486.40 | 489.35 | 491.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 484.70 | 489.35 | 491.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 485.50 | 485.76 | 488.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:15:00 | 485.05 | 486.09 | 488.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 485.75 | 486.42 | 487.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 488.50 | 486.83 | 487.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 485.20 | 486.83 | 487.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 484.00 | 486.27 | 487.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:30:00 | 480.25 | 482.96 | 484.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 460.46 | 469.45 | 475.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 461.22 | 469.45 | 475.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 460.80 | 469.45 | 475.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 461.46 | 469.45 | 475.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 456.24 | 465.71 | 472.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 456.25 | 455.90 | 463.24 | SL hit (close>ema200) qty=0.50 sl=455.90 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 478.75 | 466.90 | 465.72 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 459.00 | 464.89 | 465.35 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 467.45 | 465.83 | 465.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 473.30 | 467.83 | 466.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 466.00 | 469.00 | 467.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 466.00 | 469.00 | 467.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 466.00 | 469.00 | 467.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 466.00 | 469.00 | 467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 469.40 | 469.08 | 467.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 470.90 | 469.08 | 467.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:00:00 | 476.00 | 470.26 | 468.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 472.00 | 470.97 | 469.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 472.45 | 490.70 | 492.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 472.45 | 490.70 | 492.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 464.50 | 485.46 | 489.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 418.80 | 410.46 | 422.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 10:00:00 | 418.80 | 410.46 | 422.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 406.70 | 403.19 | 406.27 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 412.60 | 407.54 | 407.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 423.40 | 412.91 | 410.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 427.00 | 427.42 | 422.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:15:00 | 431.75 | 427.42 | 422.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:15:00 | 453.34 | 444.36 | 438.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 445.00 | 449.32 | 443.92 | SL hit (close<ema200) qty=0.50 sl=449.32 alert=retest1 |

### Cycle 14 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 429.85 | 440.10 | 441.02 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 441.50 | 439.26 | 439.09 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 435.75 | 438.59 | 438.85 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 440.05 | 438.83 | 438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 443.75 | 439.81 | 439.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 438.50 | 440.47 | 439.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 438.50 | 440.47 | 439.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 438.50 | 440.47 | 439.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 438.50 | 440.47 | 439.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 439.30 | 440.24 | 439.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 441.40 | 440.24 | 439.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 437.50 | 439.24 | 439.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 437.50 | 439.24 | 439.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 11:15:00 | 435.50 | 437.18 | 438.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 12:15:00 | 435.65 | 435.36 | 436.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 435.65 | 435.36 | 436.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 435.65 | 435.36 | 436.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 435.65 | 435.36 | 436.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 435.55 | 435.40 | 436.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:45:00 | 435.30 | 435.40 | 436.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 437.35 | 435.79 | 436.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 437.35 | 435.79 | 436.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 438.60 | 436.35 | 436.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 442.05 | 436.35 | 436.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 439.30 | 436.94 | 436.87 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 435.20 | 436.75 | 436.85 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 15:15:00 | 437.60 | 436.92 | 436.92 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 430.90 | 435.72 | 436.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 420.50 | 429.66 | 432.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 424.45 | 423.55 | 427.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 428.55 | 423.55 | 427.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 426.30 | 424.10 | 427.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 428.65 | 424.10 | 427.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 422.70 | 420.70 | 422.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 424.70 | 420.70 | 422.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 423.40 | 421.61 | 422.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:15:00 | 427.00 | 421.61 | 422.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 432.55 | 423.80 | 423.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 455.55 | 430.15 | 426.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 458.50 | 461.52 | 450.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 458.50 | 461.52 | 450.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 452.35 | 457.51 | 452.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 452.35 | 457.51 | 452.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 454.00 | 456.81 | 452.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 448.55 | 456.81 | 452.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 444.10 | 454.27 | 451.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 444.10 | 454.27 | 451.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 443.85 | 452.18 | 451.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 444.05 | 452.18 | 451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 443.70 | 449.23 | 449.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 440.00 | 446.18 | 448.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 453.95 | 447.02 | 448.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 453.95 | 447.02 | 448.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 453.95 | 447.02 | 448.31 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 452.70 | 449.26 | 449.17 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 444.30 | 448.31 | 448.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 15:15:00 | 444.00 | 447.45 | 448.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 444.60 | 443.48 | 445.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 444.60 | 443.48 | 445.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 444.60 | 443.48 | 445.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:45:00 | 444.55 | 443.48 | 445.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 441.75 | 443.31 | 445.06 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 465.45 | 446.59 | 444.85 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 441.30 | 450.48 | 451.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 434.45 | 445.76 | 448.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 424.25 | 422.99 | 427.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 424.25 | 422.99 | 427.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 409.95 | 408.86 | 411.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 409.00 | 409.30 | 411.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:30:00 | 408.10 | 409.30 | 411.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 408.85 | 409.30 | 411.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 10:15:00 | 413.55 | 410.44 | 411.17 | SL hit (close>static) qty=1.00 sl=413.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 13:15:00 | 413.30 | 411.97 | 411.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 416.50 | 413.00 | 412.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 10:15:00 | 412.80 | 414.86 | 413.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 10:15:00 | 412.80 | 414.86 | 413.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 412.80 | 414.86 | 413.92 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 412.50 | 413.57 | 413.64 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 414.80 | 413.81 | 413.75 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 412.40 | 413.53 | 413.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 409.75 | 412.69 | 413.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 407.55 | 404.23 | 406.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 407.55 | 404.23 | 406.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 407.55 | 404.23 | 406.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 410.85 | 404.23 | 406.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 406.00 | 404.58 | 406.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 408.40 | 404.58 | 406.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 406.80 | 405.03 | 406.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 404.60 | 405.02 | 405.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 384.37 | 391.33 | 394.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 399.50 | 386.34 | 389.25 | SL hit (close>ema200) qty=0.50 sl=386.34 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 420.20 | 396.55 | 393.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 434.00 | 408.87 | 400.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 500.00 | 510.38 | 495.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 500.00 | 510.38 | 495.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 511.60 | 517.73 | 505.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 520.70 | 512.89 | 505.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 518.95 | 510.31 | 507.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:45:00 | 515.35 | 512.23 | 508.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 11:15:00 | 572.77 | 546.83 | 531.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 538.65 | 547.67 | 548.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 528.50 | 542.07 | 545.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 14:15:00 | 540.00 | 536.37 | 540.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 14:15:00 | 540.00 | 536.37 | 540.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 540.00 | 536.37 | 540.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 15:00:00 | 540.00 | 536.37 | 540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 548.80 | 538.96 | 541.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 548.80 | 538.96 | 541.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 547.95 | 540.76 | 541.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 555.70 | 540.76 | 541.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 550.85 | 543.94 | 543.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 13:15:00 | 559.10 | 547.84 | 545.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 558.50 | 564.37 | 557.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 558.50 | 564.37 | 557.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 558.50 | 564.37 | 557.72 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 544.30 | 554.26 | 554.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 543.20 | 552.05 | 553.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 537.00 | 532.08 | 538.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 537.00 | 532.08 | 538.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 537.00 | 532.08 | 538.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 543.90 | 532.08 | 538.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 535.00 | 532.66 | 538.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 536.55 | 532.66 | 538.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 540.00 | 534.13 | 538.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:45:00 | 542.00 | 534.13 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 539.00 | 535.10 | 538.26 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 15:15:00 | 551.60 | 542.18 | 540.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 566.00 | 549.67 | 545.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 565.55 | 571.90 | 565.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 565.55 | 571.90 | 565.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 565.55 | 571.90 | 565.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 565.55 | 571.90 | 565.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 569.20 | 571.36 | 565.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 576.30 | 570.28 | 567.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 11:30:00 | 576.45 | 577.97 | 576.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:00:00 | 574.10 | 577.97 | 576.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 13:15:00 | 569.95 | 575.25 | 575.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 569.95 | 575.25 | 575.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 543.50 | 567.24 | 571.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 12:15:00 | 555.15 | 553.35 | 556.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 13:00:00 | 555.15 | 553.35 | 556.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 553.45 | 553.37 | 556.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 556.15 | 553.37 | 556.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 556.30 | 553.95 | 556.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 556.30 | 553.95 | 556.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 557.00 | 554.56 | 556.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 560.95 | 554.56 | 556.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 551.95 | 554.04 | 556.11 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 583.75 | 560.47 | 558.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 590.90 | 575.59 | 567.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 598.00 | 599.35 | 590.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 587.05 | 599.35 | 590.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 582.70 | 596.02 | 589.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 582.70 | 596.02 | 589.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 585.35 | 593.88 | 589.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:30:00 | 598.80 | 594.99 | 590.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 596.25 | 601.56 | 601.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 596.25 | 601.56 | 601.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 593.15 | 598.71 | 600.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 604.05 | 597.09 | 598.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 604.05 | 597.09 | 598.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 604.05 | 597.09 | 598.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 608.00 | 597.09 | 598.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 614.50 | 600.57 | 600.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 12:15:00 | 624.20 | 605.30 | 602.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 606.15 | 610.76 | 606.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 606.15 | 610.76 | 606.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 606.15 | 610.76 | 606.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 606.15 | 610.76 | 606.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 605.65 | 609.74 | 606.34 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 587.60 | 601.99 | 603.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 585.55 | 596.61 | 600.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 12:15:00 | 602.85 | 595.80 | 598.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 12:15:00 | 602.85 | 595.80 | 598.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 602.85 | 595.80 | 598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 602.85 | 595.80 | 598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 603.10 | 597.26 | 599.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:15:00 | 609.35 | 597.26 | 599.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 604.70 | 598.75 | 599.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:30:00 | 609.35 | 598.75 | 599.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 607.40 | 601.62 | 600.98 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 598.25 | 600.69 | 600.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 596.00 | 599.54 | 600.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 12:15:00 | 592.40 | 591.47 | 594.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 13:00:00 | 592.40 | 591.47 | 594.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 601.90 | 593.56 | 594.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 601.90 | 593.56 | 594.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 595.45 | 593.94 | 595.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 586.35 | 594.74 | 595.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 589.05 | 585.91 | 588.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 608.90 | 591.93 | 590.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 608.90 | 591.93 | 590.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 641.05 | 601.76 | 594.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 624.55 | 627.37 | 620.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 624.55 | 627.37 | 620.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 624.65 | 626.71 | 622.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 643.85 | 626.71 | 622.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 610.75 | 623.52 | 621.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 610.75 | 623.52 | 621.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 598.75 | 618.57 | 619.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 595.45 | 610.08 | 614.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 613.95 | 609.21 | 613.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 613.95 | 609.21 | 613.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 613.95 | 609.21 | 613.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 612.75 | 609.21 | 613.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 612.80 | 609.93 | 613.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 612.80 | 609.93 | 613.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 609.60 | 609.87 | 613.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 602.75 | 613.06 | 613.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 607.50 | 607.73 | 609.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 572.61 | 594.33 | 601.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 577.12 | 594.33 | 601.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 546.75 | 563.74 | 578.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 505.10 | 453.71 | 450.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 531.80 | 496.97 | 476.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 525.00 | 530.01 | 518.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 525.00 | 530.01 | 518.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 535.80 | 531.17 | 520.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 538.95 | 531.49 | 522.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:45:00 | 556.45 | 535.62 | 527.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:15:00 | 538.70 | 548.40 | 545.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 526.00 | 541.72 | 542.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 526.00 | 541.72 | 542.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 523.65 | 538.10 | 540.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 539.10 | 536.57 | 538.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 539.10 | 536.57 | 538.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 539.10 | 536.57 | 538.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 539.10 | 536.57 | 538.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 532.50 | 535.75 | 538.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 524.00 | 535.75 | 538.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:15:00 | 497.80 | 512.08 | 522.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-10 14:15:00 | 520.25 | 509.80 | 518.26 | SL hit (close>ema200) qty=0.50 sl=509.80 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 447.10 | 441.36 | 440.70 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 432.05 | 438.78 | 439.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 430.25 | 436.22 | 438.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 10:15:00 | 431.20 | 430.73 | 434.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:00:00 | 431.20 | 430.73 | 434.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 429.45 | 430.47 | 433.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 439.20 | 430.47 | 433.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 433.00 | 429.33 | 431.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 433.00 | 429.33 | 431.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 428.30 | 429.12 | 431.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 425.80 | 429.12 | 431.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:30:00 | 427.00 | 428.02 | 430.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 437.95 | 428.90 | 430.07 | SL hit (close>static) qty=1.00 sl=433.80 alert=retest2 |

### Cycle 51 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 432.60 | 416.97 | 416.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 445.45 | 432.94 | 428.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 437.50 | 443.87 | 439.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 437.50 | 443.87 | 439.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 437.50 | 443.87 | 439.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 437.50 | 443.87 | 439.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 436.90 | 442.47 | 439.44 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 424.40 | 435.83 | 437.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 423.80 | 433.42 | 435.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 420.85 | 419.81 | 423.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 420.85 | 419.81 | 423.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 420.85 | 419.81 | 423.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:30:00 | 412.20 | 418.70 | 422.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 411.40 | 418.70 | 422.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 412.00 | 417.36 | 421.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 438.10 | 421.68 | 421.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 438.10 | 421.68 | 421.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 458.15 | 441.14 | 432.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 467.75 | 467.87 | 458.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 472.05 | 467.87 | 458.80 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 465.95 | 468.57 | 462.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:30:00 | 464.05 | 468.57 | 462.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 470.90 | 476.70 | 472.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 470.90 | 476.70 | 472.37 | SL hit (close<ema400) qty=1.00 sl=472.37 alert=retest1 |

### Cycle 54 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 466.70 | 473.44 | 474.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 463.35 | 471.43 | 473.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 473.85 | 471.40 | 472.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 473.85 | 471.40 | 472.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 473.85 | 471.40 | 472.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 473.85 | 471.40 | 472.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 466.90 | 470.50 | 472.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 478.90 | 470.50 | 472.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 475.45 | 471.49 | 472.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 474.55 | 471.49 | 472.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 473.90 | 467.03 | 466.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 473.90 | 467.03 | 466.66 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 446.90 | 464.13 | 465.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 445.80 | 456.13 | 461.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 431.35 | 423.29 | 434.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 431.35 | 423.29 | 434.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 431.35 | 423.29 | 434.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 424.55 | 431.50 | 434.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 458.70 | 437.25 | 435.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 458.70 | 437.25 | 435.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 466.20 | 449.85 | 442.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 10:15:00 | 499.45 | 500.72 | 494.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 11:00:00 | 499.45 | 500.72 | 494.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 498.00 | 500.18 | 494.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 507.90 | 497.97 | 495.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 491.10 | 510.71 | 512.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 491.10 | 510.71 | 512.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 10:15:00 | 485.20 | 493.70 | 501.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 468.00 | 467.34 | 472.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 471.45 | 467.34 | 472.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 472.85 | 468.44 | 472.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 472.70 | 468.44 | 472.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 472.25 | 469.21 | 472.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 471.10 | 469.21 | 472.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 470.30 | 469.42 | 472.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 469.75 | 469.38 | 472.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:00:00 | 469.20 | 469.38 | 472.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 473.00 | 470.82 | 472.31 | SL hit (close>static) qty=1.00 sl=472.95 alert=retest2 |

### Cycle 59 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 500.75 | 473.67 | 471.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 504.55 | 483.50 | 476.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 487.60 | 490.31 | 483.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 487.60 | 490.31 | 483.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 479.25 | 488.10 | 483.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 479.25 | 488.10 | 483.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 479.50 | 486.38 | 483.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 469.95 | 486.38 | 483.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 474.70 | 479.88 | 480.58 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 517.35 | 485.27 | 482.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 536.60 | 495.53 | 487.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 524.90 | 530.48 | 530.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 523.90 | 529.16 | 530.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 532.30 | 528.56 | 528.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 539.45 | 534.12 | 531.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 537.95 | 539.25 | 536.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:15:00 | 534.90 | 539.25 | 536.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 534.90 | 538.38 | 536.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 540.30 | 538.38 | 536.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:30:00 | 539.60 | 539.96 | 537.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 542.35 | 546.35 | 544.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:00:00 | 538.25 | 544.73 | 544.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 537.35 | 543.25 | 543.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 537.35 | 543.25 | 543.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 531.95 | 539.91 | 541.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 14:15:00 | 540.30 | 539.99 | 541.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 540.30 | 539.99 | 541.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 543.20 | 540.79 | 541.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:45:00 | 537.40 | 539.65 | 540.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 536.30 | 539.27 | 540.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 536.75 | 539.02 | 540.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:45:00 | 537.95 | 538.64 | 539.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 538.20 | 538.56 | 539.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 539.75 | 538.56 | 539.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 541.00 | 539.04 | 539.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 561.25 | 539.04 | 539.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 553.90 | 542.02 | 540.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 553.90 | 542.02 | 540.96 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 538.00 | 540.82 | 541.05 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 545.65 | 541.79 | 541.47 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 537.85 | 540.96 | 541.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 533.60 | 539.45 | 540.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 536.75 | 536.71 | 538.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 536.75 | 536.71 | 538.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 537.60 | 536.53 | 538.06 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 540.00 | 538.70 | 538.62 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 538.00 | 538.57 | 538.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 535.00 | 537.78 | 538.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 545.40 | 538.88 | 538.51 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 538.15 | 540.60 | 540.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 537.05 | 539.89 | 540.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 536.30 | 536.18 | 538.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 538.85 | 536.72 | 538.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 536.50 | 536.72 | 538.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 534.95 | 536.36 | 537.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 531.95 | 536.36 | 537.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 524.50 | 521.60 | 524.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 532.95 | 525.55 | 524.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 532.95 | 525.55 | 524.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 536.15 | 527.67 | 525.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 524.00 | 527.84 | 526.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 526.10 | 527.49 | 526.25 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 520.35 | 525.53 | 525.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 517.15 | 523.85 | 524.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 515.95 | 505.69 | 508.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 519.50 | 508.46 | 509.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 519.70 | 508.46 | 509.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 521.25 | 512.13 | 511.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 523.85 | 514.47 | 512.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 533.00 | 535.60 | 528.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 533.00 | 535.60 | 528.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 613.35 | 608.22 | 601.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 619.15 | 612.66 | 606.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 625.05 | 614.34 | 610.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-10 09:15:00 | 681.07 | 645.00 | 631.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 691.10 | 700.73 | 701.01 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 705.40 | 701.67 | 701.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 712.70 | 704.25 | 702.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 708.05 | 705.18 | 703.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 710.75 | 706.30 | 704.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 715.75 | 707.65 | 705.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 702.45 | 705.05 | 705.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 702.45 | 705.05 | 705.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 14:15:00 | 694.00 | 702.84 | 704.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 712.80 | 705.49 | 705.18 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 696.05 | 703.60 | 704.35 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 707.25 | 704.89 | 704.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 716.25 | 707.16 | 705.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 770.80 | 755.13 | 750.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 745.35 | 752.81 | 753.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 745.35 | 752.81 | 753.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 727.90 | 742.70 | 748.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 719.90 | 719.40 | 728.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 712.35 | 718.44 | 724.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 712.35 | 718.44 | 724.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 719.35 | 718.44 | 724.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 720.55 | 714.64 | 719.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 706.20 | 715.45 | 718.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:45:00 | 711.00 | 711.14 | 715.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 710.65 | 711.44 | 714.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 711.85 | 713.03 | 714.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 670.89 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 675.45 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 675.12 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 676.26 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 692.40 | 688.64 | 699.12 | SL hit (close>ema200) qty=0.50 sl=688.64 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 714.15 | 699.26 | 698.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 715.70 | 702.55 | 700.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 715.55 | 720.06 | 714.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 732.20 | 722.49 | 716.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 735.80 | 724.01 | 719.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 738.35 | 727.61 | 721.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 730.00 | 741.69 | 742.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 730.00 | 741.69 | 742.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 753.95 | 742.37 | 741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 763.15 | 746.53 | 743.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 740.00 | 750.02 | 747.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 729.50 | 745.92 | 745.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 727.50 | 745.92 | 745.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 740.05 | 744.74 | 745.00 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 758.65 | 747.73 | 746.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 793.10 | 763.52 | 756.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 839.20 | 840.45 | 818.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 837.55 | 840.45 | 818.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 853.75 | 856.60 | 842.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 864.00 | 854.77 | 851.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 838.85 | 849.43 | 850.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 838.85 | 849.43 | 850.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 835.00 | 846.54 | 848.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 814.50 | 813.86 | 824.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 12:15:00 | 815.55 | 815.29 | 822.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 815.55 | 815.29 | 822.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 829.20 | 815.29 | 822.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 808.80 | 814.23 | 819.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 804.60 | 812.79 | 818.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 805.45 | 812.79 | 818.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 804.95 | 811.45 | 817.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 804.25 | 803.33 | 808.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 810.45 | 804.96 | 808.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:45:00 | 814.85 | 804.96 | 808.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 810.00 | 805.97 | 808.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 812.20 | 807.71 | 809.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 816.00 | 809.37 | 809.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 813.95 | 810.29 | 810.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 813.95 | 810.29 | 810.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 824.30 | 813.76 | 811.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 821.70 | 822.18 | 817.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:45:00 | 821.40 | 822.18 | 817.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 814.90 | 820.35 | 817.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 814.90 | 820.35 | 817.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 819.00 | 820.08 | 817.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 825.50 | 820.08 | 817.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 820.00 | 821.95 | 820.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 826.00 | 825.49 | 822.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 822.80 | 840.24 | 838.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 826.25 | 837.44 | 837.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 826.25 | 837.44 | 837.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 813.35 | 832.62 | 835.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 786.50 | 780.81 | 793.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 790.45 | 780.81 | 793.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 796.55 | 785.11 | 793.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 822.85 | 785.11 | 793.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 808.95 | 789.88 | 795.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:45:00 | 811.25 | 789.88 | 795.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 810.70 | 794.04 | 796.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 810.70 | 794.04 | 796.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 808.70 | 799.82 | 798.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 830.70 | 813.01 | 806.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 817.05 | 817.28 | 810.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 817.30 | 817.28 | 810.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 812.45 | 816.60 | 811.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 813.15 | 816.60 | 811.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 816.20 | 816.52 | 812.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 829.15 | 817.02 | 812.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 823.45 | 832.18 | 832.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 823.45 | 832.18 | 832.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 820.05 | 829.75 | 831.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 837.95 | 829.50 | 831.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 838.95 | 831.39 | 831.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 838.95 | 831.39 | 831.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 848.00 | 834.71 | 833.37 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 829.40 | 834.20 | 834.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 828.10 | 831.84 | 833.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 796.65 | 796.42 | 804.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:45:00 | 800.70 | 796.42 | 804.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 796.50 | 793.78 | 800.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 804.55 | 793.78 | 800.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 796.55 | 784.13 | 787.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 796.55 | 784.13 | 787.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 796.00 | 786.50 | 788.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 778.90 | 786.50 | 788.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 768.30 | 772.54 | 778.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 763.65 | 772.30 | 775.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 791.90 | 765.76 | 768.46 | SL hit (close>static) qty=1.00 sl=788.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 804.70 | 773.55 | 771.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 811.50 | 794.69 | 784.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 810.10 | 810.82 | 800.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 807.00 | 810.82 | 800.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 813.50 | 816.94 | 810.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 818.85 | 817.24 | 811.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:45:00 | 816.40 | 817.11 | 812.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 797.10 | 812.45 | 810.98 | SL hit (close<static) qty=1.00 sl=810.25 alert=retest2 |

### Cycle 96 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 798.05 | 809.57 | 809.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 790.05 | 805.67 | 808.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 789.95 | 793.10 | 796.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:30:00 | 790.70 | 793.13 | 795.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 790.85 | 792.20 | 795.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 799.95 | 793.69 | 793.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 799.95 | 793.69 | 793.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 816.95 | 807.37 | 801.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 14:15:00 | 827.80 | 831.98 | 818.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 827.80 | 831.98 | 818.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 880.45 | 893.84 | 888.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 880.45 | 893.84 | 888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 887.65 | 892.60 | 888.36 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 875.30 | 885.75 | 886.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 873.65 | 883.33 | 885.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 879.35 | 877.19 | 881.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 876.60 | 877.08 | 880.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 880.20 | 877.08 | 880.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 876.85 | 877.03 | 880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 876.85 | 877.03 | 880.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 873.65 | 875.57 | 879.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 867.60 | 873.61 | 877.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 869.00 | 872.69 | 877.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 868.05 | 872.06 | 876.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:45:00 | 865.55 | 869.11 | 873.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:15:00 | 825.55 | 840.42 | 852.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:15:00 | 824.65 | 840.42 | 852.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 13:15:00 | 824.22 | 837.02 | 850.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 822.27 | 833.39 | 847.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 10:15:00 | 836.85 | 830.70 | 842.29 | SL hit (close>ema200) qty=0.50 sl=830.70 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 856.75 | 843.36 | 842.57 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 821.60 | 840.09 | 842.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 818.95 | 835.87 | 840.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 841.35 | 830.44 | 835.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 837.85 | 831.92 | 835.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 836.00 | 831.92 | 835.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 820.00 | 829.53 | 834.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 817.20 | 825.45 | 831.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 776.34 | 786.25 | 794.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 12:15:00 | 735.48 | 747.79 | 764.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 736.50 | 721.07 | 719.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 738.45 | 729.72 | 724.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 717.95 | 728.94 | 725.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 727.00 | 728.55 | 725.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 719.70 | 728.55 | 725.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 728.90 | 728.62 | 725.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 732.65 | 728.93 | 726.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 731.45 | 729.87 | 727.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 735.35 | 730.33 | 727.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 732.35 | 732.32 | 729.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 734.75 | 733.06 | 730.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 721.60 | 731.40 | 731.07 | SL hit (close<static) qty=1.00 sl=725.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 718.30 | 728.78 | 729.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 715.00 | 726.02 | 728.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 735.50 | 720.13 | 723.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 729.40 | 721.99 | 724.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:15:00 | 732.25 | 721.99 | 724.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 734.00 | 727.15 | 726.42 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 722.80 | 725.55 | 725.78 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 727.65 | 726.22 | 726.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 732.70 | 728.31 | 727.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 724.15 | 728.16 | 727.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 723.60 | 727.25 | 726.99 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 718.15 | 725.43 | 726.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 715.85 | 723.51 | 725.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 724.80 | 722.62 | 724.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 735.10 | 725.11 | 725.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 735.95 | 725.11 | 725.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 738.85 | 727.86 | 726.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 740.90 | 732.40 | 728.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 749.00 | 749.39 | 743.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 749.00 | 749.39 | 743.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 748.50 | 749.78 | 745.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 749.30 | 749.78 | 745.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 755.30 | 750.88 | 746.44 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 738.00 | 745.32 | 746.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 735.65 | 743.39 | 745.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 706.55 | 702.81 | 710.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 706.55 | 702.81 | 710.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 707.10 | 703.89 | 709.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 712.20 | 703.89 | 709.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 702.30 | 703.71 | 708.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 709.75 | 703.71 | 708.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 710.00 | 704.97 | 708.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 708.80 | 704.97 | 708.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 705.00 | 704.98 | 708.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 702.50 | 704.98 | 708.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 703.10 | 705.36 | 708.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 714.00 | 706.73 | 708.25 | SL hit (close>static) qty=1.00 sl=710.20 alert=retest2 |

### Cycle 109 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 680.15 | 663.69 | 663.40 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 660.45 | 663.73 | 664.00 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 686.95 | 663.48 | 662.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 695.75 | 673.63 | 667.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 764.00 | 769.37 | 753.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:15:00 | 791.30 | 769.37 | 753.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 11:30:00 | 777.85 | 772.39 | 758.98 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 753.50 | 768.61 | 758.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 12:15:00 | 753.50 | 768.61 | 758.48 | SL hit (close<ema400) qty=1.00 sl=758.48 alert=retest1 |

### Cycle 112 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 861.10 | 875.02 | 875.64 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 886.05 | 875.03 | 874.76 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 13:15:00 | 865.65 | 872.98 | 873.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 863.05 | 868.27 | 871.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 863.15 | 862.28 | 865.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 867.00 | 863.52 | 865.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 867.00 | 863.52 | 865.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 867.05 | 863.52 | 865.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 868.25 | 864.47 | 865.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 870.10 | 864.47 | 865.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 862.75 | 864.53 | 865.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 860.10 | 863.06 | 864.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 817.10 | 835.25 | 841.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 831.05 | 829.52 | 836.40 | SL hit (close>ema200) qty=0.50 sl=829.52 alert=retest2 |

### Cycle 115 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 854.30 | 838.37 | 837.85 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 833.00 | 837.56 | 837.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 15:15:00 | 831.80 | 836.41 | 837.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 752.80 | 748.75 | 770.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 752.80 | 748.75 | 770.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 767.10 | 751.31 | 763.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 767.10 | 751.31 | 763.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 771.00 | 755.25 | 764.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 765.90 | 755.25 | 764.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 764.15 | 758.67 | 764.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 765.65 | 760.16 | 764.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 763.75 | 763.24 | 765.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 727.60 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 725.94 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 727.37 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 725.56 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 747.25 | 732.34 | 742.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 747.25 | 732.34 | 742.30 | SL hit (close>ema200) qty=0.50 sl=732.34 alert=retest2 |

### Cycle 117 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 758.90 | 747.52 | 746.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 762.05 | 750.43 | 748.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 755.50 | 760.22 | 755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 760.00 | 760.17 | 755.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 735.70 | 760.17 | 755.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 739.00 | 755.94 | 754.16 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 748.60 | 752.63 | 752.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 728.15 | 747.36 | 750.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 731.45 | 723.51 | 731.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 743.25 | 727.82 | 731.85 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 747.15 | 734.93 | 734.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 756.80 | 739.31 | 736.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 770.65 | 774.75 | 762.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:15:00 | 774.05 | 774.75 | 762.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 765.80 | 771.60 | 763.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 765.80 | 771.60 | 763.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 756.80 | 768.64 | 763.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 756.80 | 768.64 | 763.33 | SL hit (close<ema400) qty=1.00 sl=763.33 alert=retest1 |

### Cycle 120 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 742.10 | 768.30 | 768.97 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 787.40 | 767.32 | 765.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 826.70 | 787.45 | 775.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 811.70 | 813.13 | 800.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:45:00 | 812.20 | 813.13 | 800.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 795.60 | 810.61 | 804.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 795.60 | 810.61 | 804.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 787.00 | 805.89 | 802.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 787.00 | 805.89 | 802.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 787.15 | 799.39 | 800.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 780.60 | 795.63 | 798.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 804.90 | 798.56 | 797.78 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 768.55 | 792.32 | 795.11 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 803.70 | 794.23 | 793.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 810.30 | 798.89 | 795.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 808.55 | 809.14 | 804.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:30:00 | 808.60 | 809.14 | 804.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 863.90 | 864.54 | 853.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 878.65 | 863.16 | 857.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 13:15:00 | 966.52 | 932.75 | 904.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 976.45 | 988.74 | 988.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 965.10 | 970.98 | 975.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 977.00 | 972.04 | 975.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 972.20 | 972.07 | 975.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 957.60 | 976.02 | 976.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 986.75 | 970.41 | 969.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 986.75 | 970.41 | 969.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 994.00 | 975.13 | 971.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 1041.80 | 1043.12 | 1022.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:30:00 | 1042.70 | 1043.12 | 1022.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 15:00:00 | 475.90 | 2024-06-04 10:15:00 | 462.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-05-30 11:00:00 | 476.40 | 2024-06-04 10:15:00 | 462.35 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-05-31 09:15:00 | 479.20 | 2024-06-04 10:15:00 | 462.35 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2024-06-04 09:45:00 | 478.00 | 2024-06-04 10:15:00 | 462.35 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-06-06 13:00:00 | 452.50 | 2024-06-07 09:15:00 | 473.30 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2024-06-06 14:30:00 | 452.60 | 2024-06-07 09:15:00 | 473.30 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest2 | 2024-06-19 09:30:00 | 475.55 | 2024-06-21 11:15:00 | 523.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-19 10:45:00 | 474.30 | 2024-06-21 11:15:00 | 521.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-11 10:15:00 | 484.70 | 2024-07-18 14:15:00 | 460.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 09:30:00 | 485.50 | 2024-07-18 14:15:00 | 461.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 11:15:00 | 485.05 | 2024-07-18 14:15:00 | 460.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:30:00 | 485.75 | 2024-07-18 14:15:00 | 461.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:30:00 | 480.25 | 2024-07-19 09:15:00 | 456.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:15:00 | 484.70 | 2024-07-22 09:15:00 | 456.25 | STOP_HIT | 0.50 | 5.87% |
| SELL | retest2 | 2024-07-12 09:30:00 | 485.50 | 2024-07-22 09:15:00 | 456.25 | STOP_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2024-07-12 11:15:00 | 485.05 | 2024-07-22 09:15:00 | 456.25 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2024-07-12 14:30:00 | 485.75 | 2024-07-22 09:15:00 | 456.25 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2024-07-16 13:30:00 | 480.25 | 2024-07-22 09:15:00 | 456.25 | STOP_HIT | 0.50 | 5.00% |
| BUY | retest2 | 2024-07-25 11:15:00 | 470.90 | 2024-08-05 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-07-25 13:00:00 | 476.00 | 2024-08-05 09:15:00 | 472.45 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-07-25 15:15:00 | 472.00 | 2024-08-05 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest1 | 2024-08-21 09:15:00 | 431.75 | 2024-08-23 11:15:00 | 453.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 09:15:00 | 431.75 | 2024-08-26 09:15:00 | 445.00 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2024-09-02 09:15:00 | 441.40 | 2024-09-02 10:15:00 | 437.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-10-09 13:00:00 | 409.00 | 2024-10-10 10:15:00 | 413.55 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-09 14:30:00 | 408.10 | 2024-10-10 10:15:00 | 413.55 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-09 15:00:00 | 408.85 | 2024-10-10 10:15:00 | 413.55 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-10-21 12:00:00 | 404.60 | 2024-10-25 09:15:00 | 384.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 404.60 | 2024-10-28 09:15:00 | 399.50 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2024-10-28 10:00:00 | 399.50 | 2024-10-28 11:15:00 | 420.20 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest2 | 2024-11-05 09:15:00 | 520.70 | 2024-11-07 11:15:00 | 572.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-06 09:30:00 | 518.95 | 2024-11-07 11:15:00 | 570.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-06 10:45:00 | 515.35 | 2024-11-07 11:15:00 | 566.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:45:00 | 576.30 | 2024-12-04 13:15:00 | 569.95 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-04 11:30:00 | 576.45 | 2024-12-04 13:15:00 | 569.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-12-04 12:00:00 | 574.10 | 2024-12-04 13:15:00 | 569.95 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-13 11:30:00 | 598.80 | 2024-12-18 10:15:00 | 596.25 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-12-30 09:15:00 | 586.35 | 2025-01-01 09:15:00 | 608.90 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-12-31 12:30:00 | 589.05 | 2025-01-01 09:15:00 | 608.90 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-01-08 09:15:00 | 602.75 | 2025-01-10 09:15:00 | 572.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 607.50 | 2025-01-10 09:15:00 | 577.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 602.75 | 2025-01-13 11:15:00 | 546.75 | TARGET_HIT | 0.50 | 9.29% |
| SELL | retest2 | 2025-01-09 10:45:00 | 607.50 | 2025-01-13 12:15:00 | 542.48 | TARGET_HIT | 0.50 | 10.70% |
| BUY | retest2 | 2025-02-03 09:15:00 | 538.95 | 2025-02-05 14:15:00 | 526.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-02-03 12:45:00 | 556.45 | 2025-02-05 14:15:00 | 526.00 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-02-05 13:15:00 | 538.70 | 2025-02-05 14:15:00 | 526.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-02-07 09:15:00 | 524.00 | 2025-02-10 11:15:00 | 497.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 524.00 | 2025-02-10 14:15:00 | 520.25 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-02-25 11:15:00 | 425.80 | 2025-02-27 09:15:00 | 437.95 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-02-25 13:30:00 | 427.00 | 2025-02-27 09:15:00 | 437.95 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-02-27 11:00:00 | 426.30 | 2025-03-03 09:15:00 | 404.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:00:00 | 426.30 | 2025-03-03 12:15:00 | 414.50 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-03-13 13:30:00 | 412.20 | 2025-03-18 09:15:00 | 438.10 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2025-03-13 14:15:00 | 411.40 | 2025-03-18 09:15:00 | 438.10 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2025-03-13 15:00:00 | 412.00 | 2025-03-18 09:15:00 | 438.10 | STOP_HIT | 1.00 | -6.33% |
| BUY | retest1 | 2025-03-21 09:15:00 | 472.05 | 2025-03-25 10:15:00 | 470.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-03-28 10:15:00 | 474.55 | 2025-04-03 10:15:00 | 473.90 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-04-09 09:15:00 | 424.55 | 2025-04-11 09:15:00 | 458.70 | STOP_HIT | 1.00 | -8.04% |
| BUY | retest2 | 2025-04-22 09:15:00 | 507.90 | 2025-04-25 10:15:00 | 491.10 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-05-05 12:30:00 | 469.75 | 2025-05-05 15:15:00 | 473.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-05 13:00:00 | 469.20 | 2025-05-05 15:15:00 | 473.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-06 09:15:00 | 467.15 | 2025-05-07 10:15:00 | 483.95 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-05-26 09:15:00 | 540.30 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-05-26 10:30:00 | 539.60 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-05-28 09:30:00 | 542.35 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-28 11:00:00 | 538.25 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-05-29 13:45:00 | 537.40 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-05-30 09:15:00 | 536.30 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-05-30 10:15:00 | 536.75 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-05-30 13:45:00 | 537.95 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-06-12 10:15:00 | 531.95 | 2025-06-18 10:15:00 | 532.95 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-06-17 09:45:00 | 524.50 | 2025-06-18 10:15:00 | 532.95 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-07 14:45:00 | 619.15 | 2025-07-10 09:15:00 | 681.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:15:00 | 625.05 | 2025-07-14 09:15:00 | 687.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 10:15:00 | 708.05 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-22 11:00:00 | 710.75 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-22 11:30:00 | 715.75 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-01 09:15:00 | 770.80 | 2025-08-04 10:15:00 | 745.35 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-08-08 14:45:00 | 706.20 | 2025-08-12 13:15:00 | 670.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-11 10:45:00 | 711.00 | 2025-08-12 13:15:00 | 675.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-11 13:15:00 | 710.65 | 2025-08-12 13:15:00 | 675.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-12 09:45:00 | 711.85 | 2025-08-12 13:15:00 | 676.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 14:45:00 | 706.20 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-08-11 10:45:00 | 711.00 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-08-11 13:15:00 | 710.65 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2025-08-12 09:45:00 | 711.85 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-08-14 09:30:00 | 692.85 | 2025-08-18 09:15:00 | 714.15 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-14 14:45:00 | 694.05 | 2025-08-18 09:15:00 | 714.15 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-08-21 09:45:00 | 735.80 | 2025-08-26 14:15:00 | 730.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-21 10:45:00 | 738.35 | 2025-08-26 14:15:00 | 730.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-11 10:00:00 | 864.00 | 2025-09-11 13:15:00 | 838.85 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-17 10:30:00 | 804.60 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-17 11:15:00 | 805.45 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-17 12:15:00 | 804.95 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-18 13:00:00 | 804.25 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-23 09:15:00 | 825.50 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-23 15:15:00 | 820.00 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-09-24 09:30:00 | 826.00 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-09-26 09:45:00 | 822.80 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-07 09:15:00 | 829.15 | 2025-10-09 13:15:00 | 823.45 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-27 09:15:00 | 763.65 | 2025-10-28 09:15:00 | 791.90 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-10-31 12:30:00 | 818.85 | 2025-11-03 09:15:00 | 797.10 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-10-31 14:45:00 | 816.40 | 2025-11-03 09:15:00 | 797.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-06 12:00:00 | 789.95 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-11-06 13:30:00 | 790.70 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-06 14:30:00 | 790.85 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-20 10:45:00 | 867.60 | 2025-11-24 12:15:00 | 825.55 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-11-20 12:00:00 | 869.00 | 2025-11-24 12:15:00 | 824.65 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-11-20 13:15:00 | 868.05 | 2025-11-24 13:15:00 | 824.22 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-11-21 09:45:00 | 865.55 | 2025-11-24 14:15:00 | 822.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:45:00 | 867.60 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-11-20 12:00:00 | 869.00 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2025-11-20 13:15:00 | 868.05 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-11-21 09:45:00 | 865.55 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-26 11:30:00 | 834.00 | 2025-11-26 13:15:00 | 848.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-12-01 14:00:00 | 817.20 | 2025-12-05 09:15:00 | 776.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:00:00 | 817.20 | 2025-12-08 12:15:00 | 735.48 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-23 13:30:00 | 732.65 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-23 14:45:00 | 731.45 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-24 10:15:00 | 735.35 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-24 13:30:00 | 732.35 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-14 14:15:00 | 702.50 | 2026-01-16 09:15:00 | 714.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-01-14 15:15:00 | 703.10 | 2026-01-16 09:15:00 | 714.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-16 12:45:00 | 702.50 | 2026-01-20 09:15:00 | 667.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 701.15 | 2026-01-20 09:15:00 | 666.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:30:00 | 698.20 | 2026-01-20 09:15:00 | 663.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:45:00 | 702.50 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2026-01-19 09:15:00 | 701.15 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2026-01-19 10:30:00 | 698.20 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.13% |
| BUY | retest1 | 2026-02-02 09:15:00 | 791.30 | 2026-02-02 12:15:00 | 753.50 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest1 | 2026-02-02 11:30:00 | 777.85 | 2026-02-02 12:15:00 | 753.50 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-02-11 14:30:00 | 892.60 | 2026-02-13 09:15:00 | 861.10 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-02-19 11:30:00 | 860.10 | 2026-02-24 11:15:00 | 817.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 860.10 | 2026-02-24 15:15:00 | 831.05 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2026-03-06 09:15:00 | 765.90 | 2026-03-09 09:15:00 | 727.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:30:00 | 764.15 | 2026-03-09 09:15:00 | 725.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 765.65 | 2026-03-09 09:15:00 | 727.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 763.75 | 2026-03-09 09:15:00 | 725.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 765.90 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2026-03-06 10:30:00 | 764.15 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2026-03-06 12:15:00 | 765.65 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2026-03-06 14:45:00 | 763.75 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2026-03-19 10:15:00 | 774.05 | 2026-03-19 13:15:00 | 756.80 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-20 09:15:00 | 777.35 | 2026-03-23 09:15:00 | 747.80 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-03-23 09:30:00 | 759.90 | 2026-03-23 10:15:00 | 742.10 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-04-15 09:15:00 | 878.65 | 2026-04-16 13:15:00 | 966.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 957.60 | 2026-05-04 11:15:00 | 986.75 | STOP_HIT | 1.00 | -3.04% |
