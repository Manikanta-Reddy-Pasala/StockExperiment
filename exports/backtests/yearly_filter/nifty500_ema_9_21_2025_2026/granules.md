# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 750.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 28 |
| ALERT3 | 126 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 38 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 26
- **Target hits / Stop hits / Partials:** 2 / 40 / 2
- **Avg / median % per leg:** 0.37% / -0.47%
- **Sum % (uncompounded):** 16.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 11 | 55.0% | 2 | 18 | 0 | 1.23% | 24.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 11 | 55.0% | 2 | 18 | 0 | 1.23% | 24.6% |
| SELL (all) | 24 | 7 | 29.2% | 0 | 22 | 2 | -0.35% | -8.5% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.05% | 4.2% |
| SELL @ 3rd Alert (retest2) | 20 | 4 | 20.0% | 0 | 18 | 2 | -0.63% | -12.7% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.05% | 4.2% |
| retest2 (combined) | 40 | 15 | 37.5% | 2 | 36 | 2 | 0.30% | 11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 451.50 | 446.34 | 446.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 455.75 | 448.22 | 447.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 497.85 | 497.95 | 488.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 497.85 | 497.95 | 488.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 508.30 | 512.11 | 508.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 508.25 | 512.11 | 508.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 510.50 | 511.79 | 508.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 518.65 | 511.79 | 508.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 526.40 | 527.83 | 528.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 526.40 | 527.83 | 528.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 524.35 | 527.13 | 527.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 525.45 | 524.31 | 525.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 525.45 | 524.31 | 525.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 524.80 | 524.41 | 525.65 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 530.65 | 526.70 | 526.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 535.65 | 528.49 | 527.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 532.60 | 532.65 | 530.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:15:00 | 528.95 | 532.65 | 530.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 528.20 | 531.76 | 530.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 529.85 | 531.76 | 530.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 526.60 | 530.73 | 529.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 526.60 | 530.73 | 529.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 526.00 | 529.78 | 529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 526.00 | 529.78 | 529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 529.75 | 529.63 | 529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 529.75 | 529.63 | 529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 531.00 | 529.91 | 529.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 533.30 | 529.91 | 529.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 528.90 | 529.70 | 529.46 | SL hit (close<static) qty=1.00 sl=529.25 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 532.80 | 537.05 | 537.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 530.40 | 535.72 | 536.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 512.15 | 510.87 | 518.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 512.15 | 510.87 | 518.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 512.40 | 513.44 | 517.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 509.45 | 511.85 | 515.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 483.98 | 490.52 | 497.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 487.80 | 485.55 | 491.43 | SL hit (close>ema200) qty=0.50 sl=485.55 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 498.40 | 491.90 | 491.46 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 486.95 | 491.77 | 492.38 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 498.25 | 493.86 | 493.28 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 491.25 | 493.09 | 493.12 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 496.35 | 493.28 | 493.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 496.75 | 493.97 | 493.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 496.50 | 497.45 | 496.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 499.15 | 497.45 | 496.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 497.85 | 497.53 | 496.32 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 494.75 | 495.56 | 495.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 487.90 | 493.99 | 494.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 489.25 | 489.18 | 491.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 489.25 | 489.18 | 491.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 490.25 | 489.29 | 491.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 488.10 | 490.55 | 491.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 489.05 | 490.27 | 491.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 487.85 | 490.48 | 491.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 493.05 | 491.75 | 491.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 493.05 | 491.75 | 491.67 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 490.60 | 491.75 | 491.76 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 495.30 | 492.46 | 492.08 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 486.05 | 491.82 | 492.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 478.55 | 489.17 | 491.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 483.40 | 483.28 | 486.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 485.00 | 483.28 | 486.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 474.00 | 471.47 | 474.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 474.20 | 471.47 | 474.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 475.85 | 472.35 | 474.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 476.80 | 472.35 | 474.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 474.80 | 472.84 | 474.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:15:00 | 478.25 | 472.84 | 474.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 481.05 | 474.48 | 475.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 478.85 | 474.48 | 475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 485.60 | 476.70 | 476.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 489.80 | 484.26 | 480.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 499.65 | 502.06 | 497.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 499.65 | 502.06 | 497.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 497.60 | 501.17 | 497.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 497.05 | 501.17 | 497.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 498.90 | 500.72 | 497.77 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 495.55 | 496.92 | 496.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 492.85 | 495.61 | 496.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 482.30 | 478.03 | 482.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 482.30 | 478.03 | 482.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 478.00 | 478.03 | 481.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 481.55 | 478.03 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 476.75 | 472.21 | 474.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 473.45 | 472.21 | 474.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 470.85 | 471.93 | 474.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 470.15 | 471.58 | 474.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 469.50 | 468.39 | 471.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 480.65 | 472.97 | 472.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 480.65 | 472.97 | 472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 484.00 | 475.17 | 473.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 480.60 | 487.26 | 483.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 480.60 | 487.26 | 483.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 479.75 | 485.76 | 483.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:15:00 | 480.00 | 485.76 | 483.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 479.75 | 484.56 | 482.73 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 474.05 | 480.80 | 481.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 462.65 | 476.32 | 479.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 459.25 | 456.71 | 463.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 459.70 | 456.71 | 463.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 446.75 | 441.87 | 446.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 448.50 | 441.87 | 446.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 442.45 | 441.98 | 446.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 441.15 | 441.98 | 446.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 450.75 | 442.88 | 445.20 | SL hit (close>static) qty=1.00 sl=447.70 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 450.80 | 443.90 | 443.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 457.00 | 448.34 | 445.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 459.55 | 460.52 | 456.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:45:00 | 458.80 | 460.52 | 456.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 460.45 | 460.50 | 456.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 456.25 | 460.50 | 456.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 459.10 | 459.98 | 457.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 458.85 | 459.98 | 457.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 460.25 | 460.04 | 457.77 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 457.95 | 458.49 | 458.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 453.80 | 457.55 | 458.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 458.40 | 457.54 | 457.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 458.25 | 457.54 | 457.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 459.20 | 457.87 | 458.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 459.20 | 457.87 | 458.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 461.65 | 458.84 | 458.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 463.75 | 460.98 | 460.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 459.40 | 460.81 | 460.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 459.40 | 460.81 | 460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 456.80 | 460.01 | 460.06 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 465.90 | 461.19 | 460.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 471.00 | 464.40 | 462.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 467.55 | 468.34 | 465.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 470.45 | 468.34 | 465.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 473.05 | 469.28 | 466.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 484.60 | 470.23 | 468.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 11:15:00 | 533.06 | 526.34 | 524.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 533.60 | 539.57 | 540.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 529.45 | 537.55 | 539.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 526.95 | 525.45 | 529.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 531.80 | 525.45 | 529.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 531.30 | 527.01 | 529.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 531.30 | 527.01 | 529.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 532.10 | 528.03 | 529.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 532.10 | 528.03 | 529.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 531.50 | 529.57 | 530.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 531.45 | 529.57 | 530.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 532.15 | 530.51 | 530.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 531.15 | 529.84 | 530.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 516.95 | 530.08 | 530.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 531.35 | 522.41 | 521.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 531.35 | 522.41 | 521.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 537.30 | 527.24 | 523.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 14:15:00 | 552.30 | 553.85 | 547.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 552.30 | 553.85 | 547.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 546.00 | 551.35 | 547.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 546.00 | 551.35 | 547.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 555.55 | 552.19 | 548.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 558.00 | 553.41 | 550.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 562.45 | 551.41 | 550.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 567.10 | 569.04 | 569.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 567.10 | 569.04 | 569.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 564.15 | 568.06 | 568.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 570.20 | 567.19 | 567.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 571.00 | 567.19 | 567.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 569.95 | 567.75 | 567.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 569.95 | 567.75 | 567.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 566.45 | 567.58 | 567.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 569.15 | 567.58 | 567.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 567.85 | 567.64 | 567.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:00:00 | 562.05 | 566.52 | 567.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 566.40 | 562.65 | 562.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 566.40 | 562.65 | 562.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 576.10 | 569.33 | 566.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 571.70 | 573.39 | 569.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 571.05 | 573.39 | 569.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 572.00 | 573.11 | 570.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 570.75 | 573.11 | 570.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 572.80 | 572.70 | 570.39 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 569.25 | 570.84 | 570.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 565.20 | 569.71 | 570.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 572.25 | 569.66 | 570.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 572.25 | 569.66 | 570.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 569.75 | 569.68 | 570.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 569.75 | 569.68 | 570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 571.55 | 570.05 | 570.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 571.55 | 570.05 | 570.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 571.15 | 570.27 | 570.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 570.55 | 570.27 | 570.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 573.00 | 570.82 | 570.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 573.00 | 570.82 | 570.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 15:15:00 | 576.10 | 572.36 | 571.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 572.30 | 572.35 | 571.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 572.30 | 572.35 | 571.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 568.30 | 571.54 | 571.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 568.30 | 571.54 | 571.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 567.75 | 570.78 | 570.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 566.90 | 568.96 | 569.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 554.50 | 554.22 | 558.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 552.80 | 554.22 | 558.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 552.10 | 554.03 | 557.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:15:00 | 550.70 | 554.03 | 557.74 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 550.40 | 552.05 | 555.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:30:00 | 553.10 | 552.05 | 555.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 537.20 | 545.93 | 551.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 535.00 | 542.16 | 548.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 535.00 | 540.97 | 547.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 543.30 | 540.89 | 542.98 | SL hit (close>ema400) qty=1.00 sl=542.98 alert=retest1 |

### Cycle 31 — BUY (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 14:15:00 | 556.05 | 546.39 | 545.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 562.55 | 553.33 | 550.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 10:15:00 | 555.10 | 555.16 | 551.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:30:00 | 553.00 | 555.16 | 551.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 559.65 | 555.54 | 552.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:45:00 | 561.00 | 556.62 | 553.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 552.15 | 553.72 | 553.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 552.15 | 553.72 | 553.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 550.70 | 552.68 | 553.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 540.40 | 536.78 | 541.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 540.40 | 536.78 | 541.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 543.05 | 538.03 | 541.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 543.05 | 538.03 | 541.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 540.90 | 538.60 | 541.25 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 550.95 | 544.18 | 543.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 554.05 | 548.95 | 546.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 549.00 | 549.31 | 547.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 552.80 | 550.11 | 547.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 560.00 | 566.37 | 566.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 560.00 | 566.37 | 566.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 552.90 | 562.55 | 564.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 561.15 | 560.66 | 563.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 15:00:00 | 561.15 | 560.66 | 563.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 562.35 | 561.37 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 565.90 | 561.37 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 557.20 | 560.53 | 562.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 563.85 | 560.53 | 562.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 554.65 | 551.25 | 555.64 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 560.45 | 558.15 | 557.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 564.25 | 559.37 | 558.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 556.70 | 558.83 | 558.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 556.70 | 558.83 | 558.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 556.35 | 558.34 | 558.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 558.35 | 558.34 | 558.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 12:15:00 | 556.20 | 557.91 | 557.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 556.20 | 557.91 | 557.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 552.60 | 556.59 | 557.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 561.40 | 557.56 | 557.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 561.40 | 557.56 | 557.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 562.00 | 558.44 | 558.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 566.50 | 560.06 | 558.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 571.75 | 571.86 | 568.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:30:00 | 571.40 | 571.86 | 568.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 569.75 | 571.07 | 569.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 570.20 | 571.07 | 569.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 569.15 | 570.69 | 569.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 569.15 | 570.69 | 569.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 574.45 | 571.44 | 569.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 576.70 | 574.64 | 572.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 575.95 | 574.67 | 572.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 577.00 | 574.16 | 572.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:00:00 | 577.05 | 576.25 | 574.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 581.85 | 582.55 | 579.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 580.25 | 582.55 | 579.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 587.50 | 584.26 | 581.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 581.80 | 584.26 | 581.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 608.85 | 614.75 | 611.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 608.85 | 614.75 | 611.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 606.60 | 613.12 | 610.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 606.60 | 613.12 | 610.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 604.55 | 608.65 | 609.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 604.55 | 608.65 | 609.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 597.90 | 603.67 | 606.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 600.95 | 598.88 | 600.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 608.20 | 598.88 | 600.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 611.90 | 601.48 | 601.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 612.65 | 601.48 | 601.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 612.85 | 603.76 | 602.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 619.45 | 611.58 | 607.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 15:15:00 | 615.00 | 615.21 | 612.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 610.10 | 615.21 | 612.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 611.35 | 614.44 | 612.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 610.40 | 614.44 | 612.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 610.20 | 613.59 | 611.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 610.45 | 613.59 | 611.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 611.45 | 613.16 | 611.80 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 605.55 | 610.78 | 611.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 604.25 | 609.48 | 610.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 609.70 | 609.25 | 610.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 609.70 | 609.25 | 610.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 608.25 | 609.05 | 609.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 610.20 | 609.05 | 609.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 613.50 | 609.94 | 610.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 613.50 | 609.94 | 610.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 610.75 | 610.10 | 610.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 612.30 | 610.10 | 610.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 612.90 | 610.66 | 610.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 624.70 | 613.39 | 611.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 615.00 | 615.66 | 613.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 615.35 | 615.66 | 613.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 614.80 | 615.49 | 613.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 614.30 | 615.49 | 613.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 613.00 | 614.99 | 613.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:30:00 | 614.35 | 614.99 | 613.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 610.50 | 614.09 | 613.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 613.25 | 615.18 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 610.15 | 614.17 | 613.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 610.15 | 614.17 | 613.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 610.45 | 613.43 | 613.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 604.20 | 613.43 | 613.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 608.55 | 612.45 | 612.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 606.00 | 609.90 | 611.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 610.35 | 609.99 | 611.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 614.55 | 609.99 | 611.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 610.20 | 610.03 | 611.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 609.25 | 609.10 | 610.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 578.79 | 595.42 | 602.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 595.05 | 594.20 | 600.04 | SL hit (close>ema200) qty=0.50 sl=594.20 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 571.00 | 560.42 | 560.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 579.75 | 566.14 | 563.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 564.35 | 571.38 | 567.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 564.35 | 571.38 | 567.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 563.15 | 569.73 | 567.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 571.80 | 569.73 | 567.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 12:15:00 | 564.90 | 569.69 | 570.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 564.90 | 569.69 | 570.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 558.15 | 565.27 | 567.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 566.30 | 563.65 | 566.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 566.30 | 563.65 | 566.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 568.35 | 564.59 | 566.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 568.35 | 564.59 | 566.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 574.35 | 566.54 | 567.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 574.35 | 566.54 | 567.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 574.30 | 568.09 | 567.76 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 561.40 | 567.40 | 568.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 557.85 | 565.49 | 567.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 553.45 | 549.32 | 555.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 553.45 | 549.32 | 555.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 553.00 | 550.06 | 555.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 572.80 | 550.06 | 555.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 566.85 | 553.42 | 556.14 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 569.35 | 558.99 | 558.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 570.75 | 561.34 | 559.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 577.50 | 579.11 | 573.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 577.50 | 579.11 | 573.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 567.25 | 576.81 | 574.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 567.25 | 576.81 | 574.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 563.95 | 574.23 | 573.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 562.85 | 574.23 | 573.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 560.65 | 571.52 | 572.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 557.65 | 568.74 | 571.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 574.10 | 567.85 | 569.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 574.10 | 567.85 | 569.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 583.10 | 570.90 | 571.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 583.10 | 570.90 | 571.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 578.70 | 572.46 | 571.71 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 15:15:00 | 569.00 | 570.99 | 571.22 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 580.10 | 572.81 | 572.03 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 564.90 | 574.32 | 575.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 562.45 | 571.95 | 573.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 15:15:00 | 570.00 | 569.83 | 572.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 563.55 | 569.83 | 572.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 558.00 | 563.89 | 567.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 566.20 | 564.01 | 565.29 | SL hit (close>ema400) qty=1.00 sl=565.29 alert=retest1 |

### Cycle 53 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 569.00 | 566.42 | 566.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 573.70 | 568.21 | 567.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 576.25 | 577.17 | 573.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 576.25 | 577.17 | 573.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 575.00 | 577.29 | 575.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 572.25 | 577.29 | 575.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 575.85 | 577.00 | 575.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 577.40 | 577.00 | 575.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 576.30 | 577.02 | 575.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 576.00 | 577.02 | 575.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 596.20 | 598.55 | 596.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 595.80 | 598.55 | 596.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 594.35 | 597.71 | 596.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 594.35 | 597.71 | 596.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 591.45 | 596.46 | 595.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 591.90 | 596.46 | 595.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 591.65 | 594.85 | 595.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 15:15:00 | 586.35 | 593.15 | 594.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 566.90 | 561.38 | 568.92 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 583.25 | 571.72 | 570.99 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 555.95 | 570.49 | 572.03 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 577.45 | 570.26 | 569.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 580.20 | 573.88 | 571.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 581.40 | 583.83 | 579.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 588.15 | 584.69 | 580.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 588.65 | 585.18 | 582.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 568.65 | 581.85 | 581.22 | SL hit (close<static) qty=1.00 sl=572.20 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 562.80 | 578.04 | 579.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 560.20 | 574.47 | 577.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 570.50 | 567.05 | 572.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 570.25 | 567.05 | 572.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 561.20 | 564.75 | 569.65 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 571.95 | 569.36 | 569.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 575.70 | 570.76 | 569.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 570.45 | 579.00 | 576.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 570.45 | 579.00 | 576.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 569.45 | 577.09 | 575.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 569.45 | 577.09 | 575.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 582.15 | 578.10 | 576.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 588.15 | 578.48 | 576.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 12:15:00 | 568.00 | 578.45 | 579.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 568.00 | 578.45 | 579.73 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 585.00 | 580.57 | 580.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 593.90 | 583.23 | 581.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 15:15:00 | 620.00 | 622.75 | 614.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 09:15:00 | 622.75 | 622.75 | 614.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 621.15 | 622.76 | 617.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 621.15 | 622.76 | 617.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 620.10 | 622.26 | 618.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 637.25 | 622.26 | 618.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 606.50 | 623.55 | 622.41 | SL hit (close<static) qty=1.00 sl=616.80 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 612.50 | 621.34 | 621.51 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 628.20 | 620.72 | 620.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 632.60 | 623.10 | 621.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 631.00 | 632.17 | 628.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:45:00 | 630.55 | 632.17 | 628.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 636.30 | 639.93 | 635.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 635.25 | 639.93 | 635.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 639.55 | 639.85 | 636.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 642.15 | 636.95 | 636.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 13:15:00 | 633.60 | 638.70 | 637.67 | SL hit (close<static) qty=1.00 sl=636.20 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 631.95 | 636.28 | 636.68 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 642.00 | 637.09 | 636.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 649.40 | 639.55 | 637.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 645.10 | 646.82 | 642.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 642.75 | 646.82 | 642.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 645.40 | 646.27 | 643.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 640.55 | 646.27 | 643.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 642.85 | 645.59 | 643.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:15:00 | 648.00 | 645.47 | 643.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-29 09:15:00 | 712.80 | 704.23 | 698.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 696.00 | 702.65 | 702.75 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 706.50 | 703.42 | 703.10 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 701.10 | 703.16 | 703.27 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 706.50 | 703.94 | 703.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 707.30 | 704.62 | 703.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 750.10 | 750.17 | 742.57 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:15:00 | 518.65 | 2025-06-03 10:15:00 | 526.40 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-06-09 09:15:00 | 533.30 | 2025-06-09 09:15:00 | 528.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-09 11:00:00 | 531.25 | 2025-06-12 10:15:00 | 532.80 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-06-17 10:45:00 | 509.45 | 2025-06-19 12:15:00 | 483.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:45:00 | 509.45 | 2025-06-20 11:15:00 | 487.80 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-07-02 13:15:00 | 488.10 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-02 14:15:00 | 489.05 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-03 09:15:00 | 487.85 | 2025-07-03 11:15:00 | 493.05 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-28 12:00:00 | 470.15 | 2025-07-29 12:15:00 | 480.65 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-07-29 10:15:00 | 469.50 | 2025-07-29 12:15:00 | 480.65 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-07 11:15:00 | 441.15 | 2025-08-07 14:15:00 | 450.75 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-08 10:30:00 | 440.55 | 2025-08-12 09:15:00 | 444.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-08 14:45:00 | 440.55 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-11 09:30:00 | 440.60 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-08-11 15:00:00 | 440.30 | 2025-08-12 10:15:00 | 450.80 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-08-29 09:15:00 | 484.60 | 2025-09-12 11:15:00 | 533.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-25 11:30:00 | 531.15 | 2025-10-01 10:15:00 | 531.35 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-26 09:15:00 | 516.95 | 2025-10-01 10:15:00 | 531.35 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-10-07 15:15:00 | 558.00 | 2025-10-17 15:15:00 | 567.10 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2025-10-09 09:15:00 | 562.45 | 2025-10-17 15:15:00 | 567.10 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-10-23 13:00:00 | 562.05 | 2025-10-27 12:15:00 | 566.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-03 13:15:00 | 570.55 | 2025-11-03 13:15:00 | 573.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-10 09:15:00 | 552.80 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest1 | 2025-11-10 09:45:00 | 552.10 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest1 | 2025-11-10 10:15:00 | 550.70 | 2025-11-13 11:15:00 | 543.30 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-11-11 14:00:00 | 535.00 | 2025-11-13 13:15:00 | 555.15 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-11-11 15:15:00 | 535.00 | 2025-11-13 13:15:00 | 555.15 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-11-18 14:45:00 | 561.00 | 2025-11-20 11:15:00 | 552.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-28 09:30:00 | 552.80 | 2025-12-05 09:15:00 | 560.00 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-12-10 12:15:00 | 558.35 | 2025-12-10 12:15:00 | 556.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-17 09:30:00 | 576.70 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.83% |
| BUY | retest2 | 2025-12-17 12:15:00 | 575.95 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-17 14:15:00 | 577.00 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-12-18 10:00:00 | 577.05 | 2025-12-29 13:15:00 | 604.55 | STOP_HIT | 1.00 | 4.77% |
| SELL | retest2 | 2026-01-09 11:30:00 | 609.25 | 2026-01-12 11:15:00 | 578.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 609.25 | 2026-01-12 14:15:00 | 595.05 | STOP_HIT | 0.50 | 2.33% |
| BUY | retest2 | 2026-01-27 09:15:00 | 571.80 | 2026-01-28 12:15:00 | 564.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest1 | 2026-02-13 09:15:00 | 563.55 | 2026-02-17 11:15:00 | 566.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-03-12 11:00:00 | 588.15 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-12 15:00:00 | 588.65 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-20 09:15:00 | 588.15 | 2026-03-23 12:15:00 | 568.00 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-04-01 09:15:00 | 637.25 | 2026-04-02 09:15:00 | 606.50 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2026-04-10 09:15:00 | 642.15 | 2026-04-10 13:15:00 | 633.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-04-16 14:15:00 | 648.00 | 2026-04-29 09:15:00 | 712.80 | TARGET_HIT | 1.00 | 10.00% |
