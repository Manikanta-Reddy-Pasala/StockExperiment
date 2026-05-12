# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 427.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 62 |
| ALERT1 | 41 |
| ALERT2 | 41 |
| ALERT2_SKIP | 22 |
| ALERT3 | 98 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 65 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 45
- **Target hits / Stop hits / Partials:** 6 / 63 / 7
- **Avg / median % per leg:** 0.62% / -0.87%
- **Sum % (uncompounded):** 46.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 9 | 36.0% | 3 | 22 | 0 | 0.71% | 17.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 9 | 36.0% | 3 | 22 | 0 | 0.71% | 17.7% |
| SELL (all) | 51 | 22 | 43.1% | 3 | 41 | 7 | 0.57% | 29.0% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.00% | -8.0% |
| SELL @ 3rd Alert (retest2) | 47 | 22 | 46.8% | 3 | 37 | 7 | 0.79% | 37.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.00% | -8.0% |
| retest2 (combined) | 72 | 31 | 43.1% | 6 | 59 | 7 | 0.76% | 54.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 461.15 | 466.35 | 466.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 460.40 | 464.05 | 465.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 449.60 | 452.40 | 455.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 427.12 | 436.61 | 442.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 436.30 | 431.74 | 437.53 | SL hit (close>ema200) qty=0.50 sl=431.74 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 439.95 | 425.20 | 424.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 443.75 | 428.91 | 426.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 440.40 | 441.18 | 435.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 440.40 | 441.18 | 435.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 469.85 | 477.70 | 468.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 469.95 | 477.70 | 468.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 478.60 | 477.88 | 469.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 466.30 | 477.88 | 469.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 488.00 | 494.56 | 489.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 487.60 | 494.56 | 489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 486.75 | 493.00 | 489.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 484.50 | 493.00 | 489.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 488.75 | 492.15 | 489.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 490.50 | 490.13 | 488.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 480.10 | 487.66 | 487.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 480.10 | 487.66 | 487.79 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 491.60 | 486.06 | 485.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 495.30 | 489.52 | 487.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 492.35 | 493.62 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 490.95 | 493.08 | 491.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 490.95 | 493.08 | 491.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 486.75 | 491.82 | 491.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 486.05 | 491.82 | 491.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 485.75 | 490.60 | 490.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 483.55 | 487.69 | 489.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 491.10 | 484.50 | 486.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 496.50 | 486.90 | 487.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 496.30 | 486.90 | 487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 493.00 | 488.12 | 487.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 497.00 | 492.43 | 490.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 504.25 | 506.26 | 501.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 504.25 | 506.26 | 501.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 506.90 | 506.39 | 502.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 510.65 | 507.31 | 502.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 499.10 | 504.96 | 503.51 | SL hit (close<static) qty=1.00 sl=500.90 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 501.60 | 502.44 | 502.55 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 504.00 | 502.74 | 502.66 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 501.85 | 502.56 | 502.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 497.95 | 501.64 | 502.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 490.60 | 489.27 | 492.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 490.80 | 489.57 | 492.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 486.35 | 489.56 | 492.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:45:00 | 488.85 | 488.42 | 490.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 13:45:00 | 486.70 | 487.57 | 490.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 487.10 | 487.93 | 489.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 488.90 | 488.13 | 489.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 490.00 | 488.13 | 489.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 485.50 | 487.60 | 489.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 483.35 | 487.05 | 489.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:45:00 | 484.00 | 484.41 | 487.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 482.30 | 483.20 | 486.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 496.00 | 485.98 | 486.64 | SL hit (close>static) qty=1.00 sl=493.75 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 496.75 | 488.14 | 487.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 500.85 | 490.68 | 488.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 487.40 | 493.13 | 493.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 479.95 | 489.51 | 491.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 483.40 | 481.32 | 485.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 484.00 | 481.86 | 484.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 484.00 | 481.86 | 484.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 485.00 | 482.86 | 484.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 486.10 | 482.86 | 484.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 485.60 | 483.41 | 484.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 482.70 | 483.41 | 484.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 467.95 | 475.04 | 479.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 464.35 | 472.59 | 477.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 464.40 | 468.62 | 473.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 441.13 | 455.23 | 460.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 441.18 | 455.23 | 460.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 455.95 | 455.38 | 459.75 | SL hit (close>ema200) qty=0.50 sl=455.38 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 463.20 | 461.44 | 461.26 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 457.20 | 460.55 | 460.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 454.00 | 458.03 | 459.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 451.15 | 450.14 | 453.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 448.90 | 448.56 | 451.09 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 454.75 | 452.58 | 452.40 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 450.40 | 452.28 | 452.34 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 455.75 | 453.02 | 452.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 456.60 | 453.91 | 453.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 452.60 | 453.69 | 453.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 448.75 | 452.70 | 452.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 443.80 | 449.53 | 451.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 423.10 | 435.57 | 437.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 423.15 | 428.83 | 432.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 424.60 | 427.21 | 430.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 431.65 | 429.39 | 429.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 431.65 | 429.39 | 429.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 436.90 | 430.89 | 429.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 445.20 | 446.44 | 441.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 445.20 | 446.44 | 441.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 441.45 | 445.44 | 441.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 441.45 | 445.44 | 441.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 442.05 | 444.76 | 441.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 444.70 | 442.34 | 441.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 447.40 | 443.77 | 442.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 445.00 | 444.01 | 442.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 440.85 | 443.38 | 442.59 | SL hit (close<static) qty=1.00 sl=441.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 440.00 | 442.26 | 442.42 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 443.90 | 442.59 | 442.56 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 440.20 | 442.12 | 442.36 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 450.20 | 443.46 | 442.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 453.90 | 447.34 | 445.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 460.60 | 461.63 | 457.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 459.05 | 461.63 | 457.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 460.40 | 461.39 | 458.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 465.70 | 460.98 | 460.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 464.95 | 462.41 | 461.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 465.00 | 463.04 | 461.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 459.25 | 466.07 | 466.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 459.25 | 466.07 | 466.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 457.85 | 464.42 | 465.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 13:15:00 | 408.05 | 406.46 | 412.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 408.05 | 406.46 | 412.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 402.75 | 403.33 | 406.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 401.35 | 403.45 | 406.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 399.30 | 403.67 | 405.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 400.85 | 398.01 | 400.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 399.60 | 398.01 | 400.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 397.35 | 397.88 | 400.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 393.55 | 396.72 | 398.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:00:00 | 393.25 | 396.72 | 398.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 392.30 | 393.01 | 395.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 393.00 | 393.01 | 395.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 396.00 | 390.90 | 392.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 396.00 | 390.90 | 392.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 396.55 | 392.03 | 393.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 396.55 | 392.03 | 393.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 400.80 | 393.78 | 393.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 400.80 | 393.78 | 393.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 12:15:00 | 404.70 | 400.50 | 398.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 408.30 | 409.66 | 406.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 10:00:00 | 408.30 | 409.66 | 406.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 405.40 | 408.44 | 406.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 405.30 | 408.44 | 406.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 411.30 | 409.02 | 406.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 413.15 | 409.02 | 406.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 413.15 | 409.84 | 407.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 412.20 | 414.58 | 411.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 416.80 | 415.54 | 412.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 423.30 | 424.91 | 422.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 423.55 | 424.91 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 426.50 | 425.23 | 422.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-27 15:15:00 | 421.90 | 422.96 | 423.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 421.90 | 422.96 | 423.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 420.15 | 422.40 | 422.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 414.80 | 417.58 | 418.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 419.60 | 417.92 | 418.66 | SL hit (close>static) qty=1.00 sl=419.50 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 15:15:00 | 420.40 | 418.63 | 418.50 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 411.65 | 417.23 | 417.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 410.05 | 415.80 | 417.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 414.00 | 414.48 | 416.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 411.50 | 414.19 | 415.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:00:00 | 409.80 | 413.07 | 414.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 406.40 | 406.67 | 409.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 407.75 | 404.38 | 404.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 407.75 | 404.38 | 404.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 408.95 | 405.75 | 404.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 409.40 | 411.07 | 408.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 408.80 | 410.62 | 408.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 411.00 | 410.37 | 408.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 411.85 | 410.38 | 409.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 404.50 | 411.21 | 411.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 404.50 | 411.21 | 411.34 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 418.05 | 411.41 | 410.84 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 407.60 | 411.81 | 412.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 402.15 | 409.22 | 410.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 407.50 | 407.41 | 409.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:30:00 | 407.55 | 407.41 | 409.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 409.10 | 408.00 | 409.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:30:00 | 410.00 | 408.00 | 409.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 408.00 | 408.00 | 409.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 405.80 | 408.00 | 409.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 405.55 | 405.42 | 405.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 405.55 | 405.42 | 405.42 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 404.65 | 405.27 | 405.35 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 407.35 | 405.68 | 405.53 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 404.45 | 405.32 | 405.39 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 409.95 | 406.20 | 405.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 416.45 | 410.10 | 407.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 412.30 | 413.16 | 410.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:00:00 | 412.30 | 413.16 | 410.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 414.50 | 412.92 | 411.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 414.35 | 412.92 | 411.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 408.15 | 412.64 | 411.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 408.15 | 412.64 | 411.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 408.25 | 411.76 | 411.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:15:00 | 408.15 | 411.76 | 411.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 406.50 | 410.24 | 410.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 402.10 | 408.12 | 409.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 406.20 | 405.98 | 408.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 11:45:00 | 401.70 | 405.62 | 407.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 13:30:00 | 401.10 | 405.26 | 407.18 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 407.90 | 405.79 | 407.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 407.90 | 405.79 | 407.24 | SL hit (close>ema400) qty=1.00 sl=407.24 alert=retest1 |

### Cycle 38 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 388.45 | 383.89 | 383.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 12:15:00 | 390.30 | 385.76 | 384.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 391.00 | 394.93 | 392.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 391.75 | 394.29 | 392.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 391.75 | 394.29 | 392.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 392.30 | 393.89 | 392.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:45:00 | 395.65 | 394.37 | 392.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 395.50 | 395.63 | 394.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 400.15 | 402.16 | 402.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 400.15 | 402.16 | 402.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 398.45 | 401.42 | 401.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 387.80 | 385.40 | 388.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:00:00 | 387.80 | 385.40 | 388.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 390.00 | 386.32 | 388.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 390.00 | 386.32 | 388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 387.95 | 386.65 | 388.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 387.00 | 386.99 | 388.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 390.35 | 387.00 | 387.27 | SL hit (close>static) qty=1.00 sl=390.25 alert=retest2 |

### Cycle 40 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 390.50 | 387.70 | 387.57 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 380.60 | 386.28 | 386.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 378.10 | 384.64 | 386.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 378.45 | 378.16 | 381.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:45:00 | 379.15 | 378.16 | 381.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 377.10 | 378.01 | 380.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 376.80 | 378.01 | 380.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 376.20 | 377.84 | 380.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 376.50 | 378.77 | 379.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 373.55 | 377.30 | 378.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.96 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.39 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.68 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 365.00 | 363.82 | 367.05 | SL hit (close>ema200) qty=0.50 sl=363.82 alert=retest2 |

### Cycle 42 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 375.00 | 368.45 | 367.92 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 365.45 | 367.82 | 368.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 362.95 | 366.19 | 367.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 13:15:00 | 343.10 | 341.73 | 347.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 14:45:00 | 339.65 | 341.29 | 346.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 348.00 | 342.56 | 346.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 348.00 | 342.56 | 346.48 | SL hit (close>ema400) qty=1.00 sl=346.48 alert=retest1 |

### Cycle 44 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 354.40 | 346.02 | 345.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 357.20 | 351.35 | 348.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 349.75 | 351.03 | 348.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:30:00 | 350.95 | 351.03 | 348.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 349.35 | 350.70 | 348.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 354.65 | 350.70 | 348.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 352.60 | 361.75 | 362.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 352.60 | 361.75 | 362.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 351.90 | 357.11 | 359.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 360.00 | 357.69 | 359.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 366.00 | 359.35 | 360.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 402.60 | 359.35 | 360.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 405.00 | 368.48 | 364.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 424.55 | 419.19 | 415.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 427.00 | 427.30 | 422.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 426.00 | 427.30 | 422.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 427.35 | 427.31 | 423.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 432.75 | 427.69 | 423.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 425.30 | 427.59 | 427.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 425.30 | 427.59 | 427.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 10:15:00 | 421.80 | 425.89 | 426.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 15:15:00 | 422.70 | 422.63 | 424.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:15:00 | 418.45 | 422.63 | 424.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 428.15 | 418.11 | 418.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 428.15 | 418.11 | 418.90 | SL hit (close>ema400) qty=1.00 sl=418.90 alert=retest1 |

### Cycle 48 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 424.40 | 420.31 | 419.82 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 417.10 | 419.31 | 419.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 410.25 | 416.97 | 418.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 405.40 | 408.16 | 410.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 406.05 | 407.02 | 409.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-02 09:15:00 | 364.86 | 401.07 | 404.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 412.00 | 388.02 | 387.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 13:15:00 | 418.40 | 397.07 | 392.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 11:15:00 | 419.45 | 419.76 | 407.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 12:00:00 | 419.45 | 419.76 | 407.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 388.30 | 412.07 | 406.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 388.30 | 412.07 | 406.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 398.80 | 409.42 | 405.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 376.05 | 409.42 | 405.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 375.25 | 402.58 | 403.08 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 390.70 | 386.99 | 386.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 09:15:00 | 399.25 | 390.26 | 388.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 387.60 | 391.55 | 389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 388.80 | 391.00 | 389.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 385.40 | 391.00 | 389.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 384.00 | 389.60 | 388.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 384.00 | 389.60 | 388.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 384.65 | 388.61 | 388.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 378.50 | 388.61 | 388.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 377.75 | 386.44 | 387.58 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 388.40 | 386.50 | 386.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 389.90 | 387.12 | 386.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 392.85 | 392.03 | 390.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 384.70 | 389.70 | 390.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 384.70 | 389.70 | 390.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 383.10 | 388.38 | 389.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 368.70 | 368.88 | 374.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 383.30 | 372.27 | 375.16 | SL hit (close>static) qty=1.00 sl=378.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 388.70 | 378.22 | 377.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 389.00 | 380.38 | 378.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 383.05 | 384.09 | 381.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 379.40 | 384.09 | 381.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 377.80 | 382.83 | 381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 377.80 | 382.83 | 381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 376.75 | 381.62 | 380.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 376.65 | 381.62 | 380.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 375.05 | 379.20 | 379.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 373.25 | 377.15 | 378.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 376.15 | 368.82 | 371.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 379.50 | 370.95 | 372.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 379.80 | 370.95 | 372.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 384.15 | 375.62 | 374.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 384.95 | 379.57 | 376.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:45:00 | 380.00 | 378.23 | 376.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 380.10 | 380.79 | 378.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 378.25 | 380.79 | 378.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 418.00 | 409.59 | 407.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 418.20 | 420.64 | 420.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 415.45 | 419.60 | 420.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 419.95 | 419.17 | 420.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 419.95 | 419.17 | 420.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 419.20 | 419.17 | 419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 419.60 | 419.17 | 419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 420.35 | 419.41 | 420.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 420.35 | 419.41 | 420.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 420.00 | 419.53 | 420.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 421.10 | 419.53 | 420.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 419.70 | 419.56 | 419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 422.30 | 419.56 | 419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 429.00 | 421.45 | 420.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 436.85 | 429.04 | 427.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 427.50 | 429.99 | 428.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 425.55 | 429.10 | 428.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 425.55 | 429.10 | 428.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 427.55 | 428.79 | 427.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 432.00 | 428.79 | 427.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 10:30:00 | 428.80 | 428.75 | 428.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 425.85 | 429.38 | 429.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 425.85 | 429.38 | 429.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 422.65 | 425.65 | 427.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 13:15:00 | 425.25 | 424.94 | 426.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 425.25 | 424.94 | 426.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 427.70 | 425.49 | 426.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 429.40 | 425.49 | 426.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 430.00 | 426.40 | 426.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 427.85 | 426.40 | 426.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 427.40 | 426.90 | 427.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:45:00 | 427.95 | 426.90 | 427.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 428.25 | 427.17 | 427.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 427.80 | 427.17 | 427.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 427.75 | 427.29 | 427.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 432.05 | 428.35 | 427.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 429.00 | 429.49 | 428.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 430.70 | 429.73 | 428.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 430.70 | 429.73 | 428.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 427.00 | 429.47 | 428.91 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-17 11:30:00 | 449.60 | 2025-06-19 11:15:00 | 427.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 449.60 | 2025-06-20 09:15:00 | 436.30 | STOP_HIT | 0.50 | 2.96% |
| BUY | retest2 | 2025-07-04 14:45:00 | 490.50 | 2025-07-07 09:15:00 | 480.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-18 11:30:00 | 510.65 | 2025-07-21 09:15:00 | 499.10 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-07-25 10:15:00 | 486.35 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-07-25 12:45:00 | 488.85 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-07-25 13:45:00 | 486.70 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-28 09:15:00 | 487.10 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-07-28 12:15:00 | 483.35 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-07-28 14:45:00 | 484.00 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-07-29 10:15:00 | 482.30 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-08-06 10:30:00 | 464.35 | 2025-08-11 09:15:00 | 441.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 09:15:00 | 464.40 | 2025-08-11 09:15:00 | 441.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 10:30:00 | 464.35 | 2025-08-11 10:15:00 | 455.95 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-08-07 09:15:00 | 464.40 | 2025-08-11 10:15:00 | 455.95 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2025-08-11 15:15:00 | 464.90 | 2025-08-12 09:15:00 | 463.20 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-08-28 09:15:00 | 423.10 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-08-29 10:00:00 | 423.15 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-08-29 12:45:00 | 424.60 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-05 15:15:00 | 444.70 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-08 09:30:00 | 447.40 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-09-08 11:00:00 | 445.00 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-08 14:15:00 | 444.95 | 2025-09-09 09:15:00 | 440.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-17 11:00:00 | 465.70 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-18 10:00:00 | 464.95 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-18 15:00:00 | 465.00 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-03 10:45:00 | 401.35 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-10-06 09:15:00 | 399.30 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-07 09:30:00 | 400.85 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-10-07 10:00:00 | 399.60 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-08 09:30:00 | 393.55 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-08 10:00:00 | 393.25 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-09 09:30:00 | 392.30 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-10-09 10:00:00 | 393.00 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-16 13:15:00 | 413.15 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-10-16 14:00:00 | 413.15 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-10-17 13:45:00 | 412.20 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-10-20 10:45:00 | 416.80 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-10-30 12:30:00 | 414.80 | 2025-10-30 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-31 10:15:00 | 414.50 | 2025-10-31 13:15:00 | 419.65 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-04 12:00:00 | 409.80 | 2025-11-11 15:15:00 | 407.75 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-11-06 15:00:00 | 406.40 | 2025-11-11 15:15:00 | 407.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-13 12:30:00 | 411.00 | 2025-11-18 09:15:00 | 404.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-14 09:15:00 | 411.85 | 2025-11-18 09:15:00 | 404.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-24 09:15:00 | 405.80 | 2025-11-27 11:15:00 | 405.55 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest1 | 2025-12-04 11:45:00 | 401.70 | 2025-12-04 14:15:00 | 407.90 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-12-04 13:30:00 | 401.10 | 2025-12-04 14:15:00 | 407.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-05 09:15:00 | 403.65 | 2025-12-09 09:15:00 | 363.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 12:45:00 | 395.65 | 2025-12-29 09:15:00 | 400.15 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-12-19 11:30:00 | 395.50 | 2025-12-29 09:15:00 | 400.15 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2026-01-01 10:45:00 | 387.00 | 2026-01-02 14:15:00 | 390.35 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-07 09:15:00 | 376.80 | 2026-01-12 09:15:00 | 357.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 376.20 | 2026-01-12 09:15:00 | 357.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 376.50 | 2026-01-12 09:15:00 | 357.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 376.80 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-01-07 10:15:00 | 376.20 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-01-08 11:00:00 | 376.50 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-01-08 13:00:00 | 373.55 | 2026-01-13 09:15:00 | 354.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:00:00 | 373.55 | 2026-01-13 09:15:00 | 364.85 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2026-01-21 14:45:00 | 339.65 | 2026-01-22 09:15:00 | 348.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-01-22 11:30:00 | 344.25 | 2026-01-22 12:15:00 | 348.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-23 09:30:00 | 344.20 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-23 10:15:00 | 344.15 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-01-23 15:00:00 | 343.75 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-28 11:15:00 | 354.65 | 2026-02-02 10:15:00 | 352.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-02-13 11:15:00 | 432.75 | 2026-02-17 14:15:00 | 425.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest1 | 2026-02-19 09:15:00 | 418.45 | 2026-02-23 09:15:00 | 428.15 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-02-26 11:30:00 | 405.40 | 2026-03-02 09:15:00 | 364.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 406.05 | 2026-03-02 09:15:00 | 365.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 11:15:00 | 392.85 | 2026-03-19 13:15:00 | 384.70 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-24 10:30:00 | 368.70 | 2026-03-24 12:15:00 | 383.30 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2026-04-02 12:45:00 | 380.00 | 2026-04-15 09:15:00 | 418.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:30:00 | 380.10 | 2026-04-15 09:15:00 | 418.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:15:00 | 378.25 | 2026-04-15 09:15:00 | 416.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 09:15:00 | 432.00 | 2026-05-05 10:15:00 | 425.85 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-30 10:30:00 | 428.80 | 2026-05-05 10:15:00 | 425.85 | STOP_HIT | 1.00 | -0.69% |
