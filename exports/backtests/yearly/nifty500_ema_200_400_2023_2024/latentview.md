# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 314.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 8 |
| ALERT3 | 78 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 59 |
| PARTIAL | 17 |
| TARGET_HIT | 12 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 44
- **Target hits / Stop hits / Partials:** 12 / 48 / 17
- **Avg / median % per leg:** 1.39% / -0.93%
- **Sum % (uncompounded):** 107.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 5 | 16.1% | 5 | 26 | 0 | -0.01% | -0.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.63% | -2.6% |
| BUY @ 3rd Alert (retest2) | 30 | 5 | 16.7% | 5 | 25 | 0 | 0.08% | 2.5% |
| SELL (all) | 46 | 28 | 60.9% | 7 | 22 | 17 | 2.33% | 107.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 28 | 60.9% | 7 | 22 | 17 | 2.33% | 107.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.63% | -2.6% |
| retest2 (combined) | 76 | 33 | 43.4% | 12 | 47 | 17 | 1.44% | 109.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 10:15:00 | 463.50 | 478.85 | 478.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 11:15:00 | 456.75 | 478.63 | 478.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 483.00 | 476.70 | 477.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 10:15:00 | 483.00 | 476.70 | 477.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 483.00 | 476.70 | 477.77 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 526.45 | 478.94 | 478.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 10:15:00 | 536.10 | 481.63 | 480.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 511.15 | 511.94 | 498.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 11:45:00 | 512.40 | 511.88 | 499.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 507.70 | 511.98 | 499.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 508.70 | 511.98 | 499.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 500.00 | 511.72 | 499.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 499.00 | 511.72 | 499.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 500.00 | 511.61 | 499.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:30:00 | 500.00 | 511.61 | 499.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 500.00 | 511.49 | 499.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 498.90 | 511.37 | 499.75 | SL hit (close<ema400) qty=1.00 sl=499.75 alert=retest1 |

### Cycle 3 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 470.00 | 494.21 | 494.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 464.00 | 493.18 | 493.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 491.25 | 489.66 | 491.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 491.25 | 489.66 | 491.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 491.25 | 489.66 | 491.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 490.95 | 489.65 | 491.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 482.15 | 489.56 | 491.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 474.85 | 488.66 | 490.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:45:00 | 473.10 | 488.52 | 490.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:00:00 | 475.00 | 488.39 | 490.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 475.00 | 488.27 | 490.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 482.00 | 485.76 | 489.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:45:00 | 486.00 | 485.76 | 489.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 480.00 | 485.27 | 488.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 461.00 | 484.83 | 488.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:00:00 | 465.60 | 484.63 | 488.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 460.00 | 484.32 | 488.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 451.11 | 482.71 | 487.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 449.44 | 482.71 | 487.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 451.25 | 482.71 | 487.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 451.25 | 482.71 | 487.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 11:00:00 | 464.00 | 482.52 | 487.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 483.00 | 481.85 | 486.66 | SL hit (close>ema200) qty=0.50 sl=481.85 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 14:15:00 | 512.50 | 490.56 | 490.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 15:15:00 | 515.00 | 490.81 | 490.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 521.00 | 521.15 | 511.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 521.00 | 521.15 | 511.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 511.25 | 520.86 | 512.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 511.40 | 520.86 | 512.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 508.65 | 520.74 | 511.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:15:00 | 510.65 | 520.74 | 511.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 511.50 | 520.44 | 511.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 511.00 | 520.44 | 511.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 510.50 | 520.35 | 511.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:45:00 | 511.00 | 520.35 | 511.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 510.10 | 520.24 | 511.95 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 494.90 | 506.06 | 506.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 490.75 | 504.22 | 505.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 14:15:00 | 494.00 | 490.81 | 497.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 15:00:00 | 494.00 | 490.81 | 497.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 495.00 | 490.85 | 497.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 490.70 | 490.85 | 497.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 12:00:00 | 489.15 | 490.37 | 496.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 490.55 | 490.39 | 496.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:00:00 | 489.95 | 490.44 | 496.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 490.55 | 487.19 | 493.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:30:00 | 491.55 | 487.19 | 493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 466.16 | 483.38 | 490.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 464.69 | 483.38 | 490.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 466.02 | 483.38 | 490.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 465.45 | 483.38 | 490.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 481.40 | 474.55 | 482.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 481.40 | 474.55 | 482.95 | SL hit (close>ema200) qty=0.50 sl=474.55 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 501.90 | 472.13 | 471.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 504.00 | 472.73 | 472.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 474.20 | 478.36 | 475.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 13:15:00 | 474.20 | 478.36 | 475.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 474.20 | 478.36 | 475.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 474.20 | 478.36 | 475.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 473.90 | 478.32 | 475.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 473.90 | 478.32 | 475.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 473.90 | 478.27 | 475.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 473.00 | 478.25 | 475.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 473.05 | 478.20 | 475.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 477.05 | 476.96 | 475.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 476.45 | 477.38 | 475.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 470.00 | 477.16 | 475.29 | SL hit (close<static) qty=1.00 sl=470.70 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 446.20 | 475.20 | 475.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 440.90 | 466.41 | 470.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 453.00 | 450.19 | 459.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:15:00 | 456.35 | 450.19 | 459.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 459.40 | 450.28 | 459.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 459.55 | 450.28 | 459.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 461.00 | 450.39 | 459.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 461.00 | 450.39 | 459.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 459.55 | 450.48 | 459.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:00:00 | 458.60 | 450.77 | 459.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 10:15:00 | 462.85 | 451.07 | 459.83 | SL hit (close>static) qty=1.00 sl=462.65 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 422.35 | 400.88 | 400.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 423.50 | 401.31 | 401.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 409.75 | 410.08 | 406.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 409.75 | 410.08 | 406.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 404.80 | 410.97 | 407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 404.80 | 410.97 | 407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 405.50 | 410.92 | 407.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 418.40 | 409.18 | 407.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 14:15:00 | 404.40 | 409.32 | 407.13 | SL hit (close<static) qty=1.00 sl=404.55 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 389.80 | 415.92 | 415.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 388.75 | 415.65 | 415.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 428.60 | 410.60 | 412.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 430.45 | 410.80 | 413.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 433.25 | 410.80 | 413.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 425.30 | 414.96 | 414.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 09:15:00 | 436.40 | 415.38 | 415.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 423.90 | 415.95 | 415.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 419.75 | 419.73 | 417.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 419.95 | 419.62 | 417.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 424.00 | 419.31 | 417.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 418.15 | 419.34 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 418.35 | 419.34 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 416.60 | 419.31 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 416.60 | 419.31 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 417.70 | 419.29 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 415.95 | 419.29 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 417.10 | 419.27 | 417.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 422.10 | 419.27 | 417.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 420.20 | 419.90 | 418.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 416.10 | 419.87 | 418.17 | SL hit (close<static) qty=1.00 sl=417.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 406.30 | 416.73 | 416.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 404.80 | 416.61 | 416.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 416.00 | 414.72 | 415.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 415.10 | 414.72 | 415.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 419.30 | 414.72 | 415.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 418.75 | 414.76 | 415.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 419.35 | 414.76 | 415.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 426.05 | 414.87 | 415.71 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 435.00 | 416.62 | 416.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 441.10 | 418.08 | 417.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 417.65 | 424.64 | 421.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 420.25 | 424.60 | 421.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 419.30 | 424.60 | 421.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 422.00 | 424.57 | 421.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:30:00 | 426.00 | 424.53 | 421.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 425.40 | 424.56 | 421.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 418.10 | 424.46 | 421.37 | SL hit (close<static) qty=1.00 sl=420.00 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 432.00 | 458.90 | 458.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 428.65 | 458.60 | 458.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 447.50 | 431.21 | 442.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 439.00 | 431.28 | 442.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 450.60 | 431.28 | 442.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 447.50 | 431.45 | 442.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 447.50 | 431.45 | 442.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 449.20 | 431.62 | 442.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 449.10 | 431.62 | 442.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 318.30 | 300.03 | 320.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 318.45 | 300.03 | 320.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-04-19 11:45:00 | 512.40 | 2024-04-23 13:15:00 | 498.90 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-04-23 14:45:00 | 508.90 | 2024-04-29 09:15:00 | 495.50 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-04-24 10:30:00 | 503.00 | 2024-04-29 09:15:00 | 495.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-04-25 11:00:00 | 501.00 | 2024-04-29 09:15:00 | 495.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-04-25 12:00:00 | 502.00 | 2024-04-29 09:15:00 | 495.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-05-28 09:45:00 | 474.85 | 2024-06-05 09:15:00 | 451.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 10:45:00 | 473.10 | 2024-06-05 09:15:00 | 449.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:00:00 | 475.00 | 2024-06-05 09:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:30:00 | 475.00 | 2024-06-05 09:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:45:00 | 474.85 | 2024-06-06 10:15:00 | 483.00 | STOP_HIT | 0.50 | -1.72% |
| SELL | retest2 | 2024-05-28 10:45:00 | 473.10 | 2024-06-06 10:15:00 | 483.00 | STOP_HIT | 0.50 | -2.09% |
| SELL | retest2 | 2024-05-28 12:00:00 | 475.00 | 2024-06-06 10:15:00 | 483.00 | STOP_HIT | 0.50 | -1.68% |
| SELL | retest2 | 2024-05-28 12:30:00 | 475.00 | 2024-06-06 10:15:00 | 483.00 | STOP_HIT | 0.50 | -1.68% |
| SELL | retest2 | 2024-06-04 09:15:00 | 461.00 | 2024-06-06 14:15:00 | 496.10 | STOP_HIT | 1.00 | -7.61% |
| SELL | retest2 | 2024-06-04 10:00:00 | 465.60 | 2024-06-06 14:15:00 | 496.10 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2024-06-04 10:30:00 | 460.00 | 2024-06-06 14:15:00 | 496.10 | STOP_HIT | 1.00 | -7.85% |
| SELL | retest2 | 2024-06-05 11:00:00 | 464.00 | 2024-06-06 14:15:00 | 496.10 | STOP_HIT | 1.00 | -6.92% |
| SELL | retest2 | 2024-09-11 09:15:00 | 490.70 | 2024-10-03 12:15:00 | 466.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 12:00:00 | 489.15 | 2024-10-03 12:15:00 | 464.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 13:15:00 | 490.55 | 2024-10-03 12:15:00 | 466.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:00:00 | 489.95 | 2024-10-03 12:15:00 | 465.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 09:15:00 | 490.70 | 2024-10-17 09:15:00 | 481.40 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2024-09-12 12:00:00 | 489.15 | 2024-10-17 09:15:00 | 481.40 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2024-09-12 13:15:00 | 490.55 | 2024-10-17 09:15:00 | 481.40 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2024-09-13 11:00:00 | 489.95 | 2024-10-17 09:15:00 | 481.40 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2024-11-14 11:00:00 | 463.55 | 2024-11-19 15:15:00 | 440.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 11:30:00 | 461.90 | 2024-11-19 15:15:00 | 438.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 11:00:00 | 463.55 | 2024-12-03 09:15:00 | 466.90 | STOP_HIT | 0.50 | -0.72% |
| SELL | retest2 | 2024-11-14 11:30:00 | 461.90 | 2024-12-03 09:15:00 | 466.90 | STOP_HIT | 0.50 | -1.08% |
| BUY | retest2 | 2024-12-27 09:15:00 | 477.05 | 2024-12-30 14:15:00 | 470.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-12-30 10:45:00 | 476.45 | 2024-12-30 14:15:00 | 470.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-12-31 11:00:00 | 477.05 | 2025-01-10 09:15:00 | 462.10 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-01-09 11:00:00 | 476.60 | 2025-01-10 09:15:00 | 462.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-02-05 15:00:00 | 458.60 | 2025-02-06 10:15:00 | 462.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-02-06 12:45:00 | 458.50 | 2025-02-11 09:15:00 | 435.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 13:30:00 | 457.60 | 2025-02-11 09:15:00 | 434.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 15:15:00 | 458.00 | 2025-02-11 09:15:00 | 435.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 453.60 | 2025-02-11 09:15:00 | 430.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 453.45 | 2025-02-11 09:15:00 | 430.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 15:00:00 | 447.85 | 2025-02-11 09:15:00 | 431.77 | PARTIAL | 0.50 | 3.59% |
| SELL | retest2 | 2025-02-10 12:15:00 | 454.50 | 2025-02-11 10:15:00 | 425.46 | PARTIAL | 0.50 | 6.39% |
| SELL | retest2 | 2025-02-06 12:45:00 | 458.50 | 2025-02-11 12:15:00 | 412.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 13:30:00 | 457.60 | 2025-02-11 12:15:00 | 411.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 15:15:00 | 458.00 | 2025-02-11 12:15:00 | 412.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 453.60 | 2025-02-11 12:15:00 | 409.05 | TARGET_HIT | 0.50 | 9.82% |
| SELL | retest2 | 2025-02-07 13:45:00 | 453.45 | 2025-02-11 13:15:00 | 408.24 | TARGET_HIT | 0.50 | 9.97% |
| SELL | retest2 | 2025-02-07 15:00:00 | 447.85 | 2025-02-11 13:15:00 | 408.11 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2025-02-10 12:15:00 | 454.50 | 2025-02-11 13:15:00 | 403.07 | TARGET_HIT | 0.50 | 11.32% |
| SELL | retest2 | 2025-04-17 14:45:00 | 392.80 | 2025-04-17 15:15:00 | 394.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-04-21 10:00:00 | 392.20 | 2025-04-21 10:15:00 | 397.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-04-22 15:00:00 | 392.90 | 2025-04-23 09:15:00 | 411.25 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-04-30 15:15:00 | 390.05 | 2025-05-02 09:15:00 | 395.45 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-05-06 09:45:00 | 390.20 | 2025-05-07 15:15:00 | 400.10 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-05-06 14:45:00 | 388.45 | 2025-05-07 15:15:00 | 400.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-05-07 11:45:00 | 390.00 | 2025-05-07 15:15:00 | 400.10 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-06-17 09:15:00 | 418.40 | 2025-06-17 14:15:00 | 404.40 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-18 09:15:00 | 407.85 | 2025-06-18 10:15:00 | 404.25 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-18 09:45:00 | 406.95 | 2025-06-18 10:15:00 | 404.25 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-18 11:30:00 | 406.80 | 2025-06-19 12:15:00 | 396.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-06-23 13:45:00 | 407.50 | 2025-07-03 14:15:00 | 448.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 14:30:00 | 408.00 | 2025-07-03 14:15:00 | 448.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 11:15:00 | 405.60 | 2025-08-01 14:15:00 | 401.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-29 12:00:00 | 406.00 | 2025-08-01 14:15:00 | 401.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-03 13:00:00 | 423.90 | 2025-09-22 10:15:00 | 416.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-12 12:00:00 | 419.75 | 2025-09-22 10:15:00 | 416.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-15 09:45:00 | 419.95 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-09-17 09:15:00 | 424.00 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-09-18 09:15:00 | 422.10 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-09-22 09:45:00 | 420.20 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-10-27 14:30:00 | 426.00 | 2025-10-28 11:15:00 | 418.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-28 10:00:00 | 425.40 | 2025-10-28 11:15:00 | 418.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-30 14:15:00 | 423.75 | 2025-11-03 09:15:00 | 466.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 09:15:00 | 424.80 | 2025-11-03 09:15:00 | 467.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-08 09:15:00 | 464.55 | 2025-12-08 15:15:00 | 511.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-02 09:15:00 | 455.80 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-01-06 09:45:00 | 455.75 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-06 12:30:00 | 455.95 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.77% |
