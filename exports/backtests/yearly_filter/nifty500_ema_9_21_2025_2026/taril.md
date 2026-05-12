# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 325.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 53 |
| ALERT2 | 51 |
| ALERT2_SKIP | 27 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 33 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 27
- **Target hits / Stop hits / Partials:** 2 / 35 / 9
- **Avg / median % per leg:** 0.91% / -0.57%
- **Sum % (uncompounded):** 42.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 2 | 14 | 1 | -0.25% | -4.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.76% | 8.8% |
| BUY @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.08% | -13.0% |
| SELL (all) | 29 | 16 | 55.2% | 0 | 21 | 8 | 1.59% | 46.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 0 | 21 | 8 | 1.59% | 46.2% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.76% | 8.8% |
| retest2 (combined) | 41 | 17 | 41.5% | 1 | 32 | 8 | 0.81% | 33.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 502.35 | 488.00 | 487.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 519.75 | 502.07 | 495.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 510.00 | 511.23 | 506.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 510.00 | 511.23 | 506.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 496.50 | 507.66 | 506.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 496.80 | 507.66 | 506.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 496.00 | 505.33 | 505.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 496.00 | 505.33 | 505.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 11:15:00 | 498.00 | 503.86 | 504.56 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 505.30 | 503.64 | 503.61 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 499.00 | 502.71 | 503.19 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 522.05 | 506.13 | 504.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 535.90 | 523.10 | 514.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 520.20 | 525.38 | 518.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 520.20 | 525.38 | 518.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 512.55 | 522.82 | 518.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 512.55 | 522.82 | 518.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 514.15 | 521.08 | 517.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 508.20 | 521.08 | 517.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 508.95 | 517.89 | 516.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 508.95 | 517.89 | 516.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 499.90 | 514.29 | 515.36 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 514.80 | 510.88 | 510.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 521.00 | 514.21 | 512.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 525.75 | 518.64 | 516.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 522.90 | 518.64 | 516.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 511.35 | 517.28 | 517.12 | SL hit (close<static) qty=1.00 sl=512.60 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 513.00 | 516.42 | 516.74 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 524.85 | 518.03 | 517.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 532.05 | 520.83 | 518.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 523.95 | 523.98 | 521.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:45:00 | 529.85 | 523.98 | 521.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 523.70 | 523.92 | 521.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 522.95 | 523.92 | 521.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 523.10 | 523.76 | 521.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 523.10 | 523.76 | 521.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 522.65 | 523.54 | 521.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 522.65 | 523.54 | 521.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 522.90 | 523.41 | 521.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 521.55 | 523.41 | 521.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 520.15 | 522.76 | 521.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 518.95 | 522.76 | 521.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 522.40 | 522.69 | 521.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 528.20 | 522.69 | 521.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 534.60 | 525.07 | 522.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:30:00 | 535.00 | 527.57 | 524.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 536.60 | 529.18 | 526.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 536.10 | 527.93 | 527.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 539.10 | 530.14 | 528.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 531.50 | 533.16 | 531.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 531.50 | 533.16 | 531.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 531.40 | 532.81 | 531.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 530.40 | 532.81 | 531.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 531.00 | 532.45 | 531.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 524.75 | 532.45 | 531.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 524.05 | 530.77 | 530.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 524.05 | 530.77 | 530.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 518.55 | 525.99 | 528.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 13:15:00 | 518.75 | 518.55 | 522.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 518.75 | 518.55 | 522.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 521.85 | 518.95 | 521.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 521.25 | 518.95 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 519.90 | 519.14 | 521.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:15:00 | 517.95 | 519.14 | 521.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:45:00 | 518.20 | 518.05 | 519.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.05 | 505.69 | 510.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.29 | 505.69 | 510.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 494.95 | 493.36 | 498.29 | SL hit (close>ema200) qty=0.50 sl=493.36 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 496.90 | 486.47 | 485.98 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 487.95 | 488.67 | 488.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 484.80 | 487.81 | 488.26 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 493.05 | 488.77 | 488.61 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 486.90 | 488.41 | 488.48 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 488.65 | 488.46 | 488.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 492.40 | 489.25 | 488.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 489.60 | 490.92 | 489.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 490.00 | 490.73 | 489.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 499.15 | 490.73 | 489.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 15:15:00 | 488.80 | 490.54 | 490.39 | SL hit (close<static) qty=1.00 sl=489.10 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 469.20 | 486.27 | 488.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 465.50 | 469.67 | 475.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 469.10 | 468.85 | 474.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 469.10 | 468.85 | 474.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 473.00 | 470.08 | 474.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 469.05 | 470.62 | 472.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 468.30 | 469.66 | 470.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 469.20 | 469.66 | 470.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 469.10 | 469.46 | 470.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 482.15 | 472.00 | 471.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 482.15 | 472.00 | 471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 14:15:00 | 487.90 | 478.22 | 474.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 511.50 | 512.09 | 502.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 12:15:00 | 514.30 | 512.09 | 502.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 501.05 | 509.76 | 505.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 501.05 | 509.76 | 505.28 | SL hit (close<ema400) qty=1.00 sl=505.28 alert=retest1 |

### Cycle 18 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 501.00 | 503.35 | 503.49 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 503.95 | 503.64 | 503.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 508.60 | 504.76 | 504.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 505.10 | 506.78 | 505.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 510.75 | 507.57 | 505.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 513.00 | 509.60 | 507.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 517.50 | 510.33 | 508.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 511.55 | 513.32 | 513.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 511.55 | 513.32 | 513.34 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 514.10 | 513.48 | 513.41 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 512.00 | 513.30 | 513.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 510.25 | 512.64 | 513.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 491.65 | 489.36 | 495.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 14:45:00 | 493.35 | 489.36 | 495.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 487.75 | 485.85 | 489.63 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 510.40 | 491.17 | 490.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 513.35 | 508.15 | 503.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 509.40 | 510.00 | 506.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:30:00 | 519.50 | 512.10 | 507.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 516.75 | 516.61 | 511.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 513.45 | 516.61 | 511.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 518.70 | 517.03 | 511.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 513.85 | 517.03 | 511.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 519.30 | 517.48 | 512.65 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 12:15:00 | 545.48 | 523.98 | 516.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:45:00 | 532.25 | 523.98 | 516.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 571.45 | 542.38 | 528.95 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 24 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 505.45 | 530.65 | 531.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 499.10 | 512.81 | 520.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 504.20 | 501.94 | 511.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 503.20 | 503.13 | 510.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 499.75 | 502.36 | 509.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 529.70 | 504.45 | 506.16 | SL hit (close>static) qty=1.00 sl=511.80 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 526.10 | 508.78 | 507.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 534.00 | 513.82 | 510.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 516.60 | 519.68 | 514.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 507.45 | 517.24 | 514.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 507.45 | 517.24 | 514.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 507.15 | 515.22 | 513.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 505.35 | 515.22 | 513.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 507.30 | 513.43 | 512.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 507.30 | 513.43 | 512.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 506.25 | 512.00 | 512.36 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 513.90 | 512.72 | 512.60 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 507.90 | 512.06 | 512.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 505.85 | 508.37 | 510.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 505.10 | 507.76 | 509.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 510.00 | 503.16 | 502.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 510.00 | 503.16 | 502.28 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 502.60 | 503.12 | 503.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 500.00 | 502.12 | 502.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 499.35 | 495.50 | 497.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 496.60 | 495.72 | 497.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 494.85 | 495.72 | 497.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 494.70 | 493.81 | 495.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 501.70 | 495.38 | 496.30 | SL hit (close>static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 504.15 | 497.14 | 497.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 507.30 | 502.44 | 500.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 500.05 | 504.79 | 503.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 500.35 | 503.90 | 503.05 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 500.90 | 502.33 | 502.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 495.15 | 500.49 | 501.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 501.90 | 499.70 | 500.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 504.00 | 500.56 | 501.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 504.00 | 500.56 | 501.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 502.60 | 500.97 | 501.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 499.20 | 500.97 | 501.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 505.85 | 497.02 | 497.20 | SL hit (close>static) qty=1.00 sl=504.15 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 509.10 | 499.43 | 498.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 519.20 | 504.93 | 501.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 508.50 | 510.73 | 505.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 508.50 | 510.73 | 505.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 505.50 | 509.68 | 505.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 505.10 | 509.68 | 505.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 501.75 | 508.10 | 505.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 501.90 | 508.10 | 505.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 501.05 | 506.69 | 504.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 501.00 | 506.69 | 504.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 505.00 | 505.64 | 504.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 501.90 | 505.64 | 504.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 499.55 | 504.42 | 504.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 498.35 | 504.42 | 504.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 498.95 | 503.33 | 503.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 498.05 | 500.72 | 502.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 503.30 | 500.70 | 501.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 506.00 | 501.76 | 502.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 506.00 | 501.76 | 502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 505.45 | 502.50 | 502.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 507.35 | 502.50 | 502.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 503.85 | 502.77 | 502.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 517.60 | 506.67 | 504.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 524.85 | 525.72 | 520.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 524.85 | 525.72 | 520.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 520.80 | 524.04 | 521.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 517.35 | 524.04 | 521.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 520.80 | 523.39 | 520.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 519.55 | 523.39 | 520.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 520.30 | 522.77 | 520.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 526.50 | 522.38 | 520.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 523.50 | 531.52 | 531.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 523.50 | 531.52 | 531.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 517.35 | 525.38 | 528.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 490.70 | 487.84 | 493.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 492.40 | 487.84 | 493.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 489.35 | 488.14 | 493.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 490.30 | 488.14 | 493.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 492.15 | 489.49 | 491.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 495.00 | 489.49 | 491.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 492.95 | 490.18 | 491.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 496.80 | 490.18 | 491.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 493.80 | 490.91 | 492.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 493.80 | 490.91 | 492.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 493.15 | 491.36 | 492.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 494.75 | 491.36 | 492.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 495.20 | 492.78 | 492.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 496.75 | 493.57 | 493.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 494.95 | 495.05 | 494.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 494.95 | 495.05 | 494.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 494.25 | 494.89 | 494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 494.25 | 494.89 | 494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 494.00 | 494.71 | 494.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 494.70 | 494.71 | 494.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 494.15 | 494.60 | 494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:15:00 | 494.90 | 494.60 | 494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 494.90 | 494.66 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 497.30 | 494.66 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 495.70 | 494.87 | 494.32 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 493.50 | 494.24 | 494.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 490.40 | 493.48 | 493.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 490.25 | 489.15 | 490.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 491.40 | 489.60 | 490.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 492.00 | 489.60 | 490.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 495.90 | 490.86 | 491.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 496.75 | 490.86 | 491.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 491.90 | 491.07 | 491.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 491.20 | 491.07 | 491.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 491.30 | 491.25 | 491.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 492.00 | 491.40 | 491.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 492.00 | 491.40 | 491.39 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 491.00 | 491.32 | 491.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 15:15:00 | 490.75 | 491.21 | 491.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 493.90 | 487.08 | 488.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 489.95 | 487.66 | 488.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 487.25 | 487.35 | 488.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 486.45 | 485.19 | 485.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 13:15:00 | 462.89 | 467.56 | 472.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 13:15:00 | 462.13 | 467.56 | 472.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 466.65 | 464.04 | 469.62 | SL hit (close>ema200) qty=0.50 sl=464.04 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 318.20 | 308.01 | 307.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 336.90 | 317.77 | 312.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 325.20 | 327.54 | 321.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 325.20 | 327.54 | 321.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 322.50 | 326.53 | 321.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 323.45 | 326.53 | 321.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 319.40 | 325.11 | 321.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 319.40 | 325.11 | 321.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 316.75 | 323.44 | 320.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 316.75 | 323.44 | 320.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 307.70 | 317.95 | 318.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 303.65 | 308.47 | 312.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 313.70 | 308.00 | 311.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 312.55 | 308.91 | 311.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 314.30 | 308.91 | 311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 312.75 | 310.01 | 311.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:45:00 | 313.45 | 310.01 | 311.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 312.05 | 310.42 | 311.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:30:00 | 313.95 | 310.42 | 311.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 305.00 | 293.68 | 297.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 305.00 | 293.68 | 297.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 303.35 | 295.61 | 297.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 299.60 | 295.61 | 297.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:15:00 | 284.62 | 290.17 | 293.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 275.20 | 274.88 | 280.85 | SL hit (close>ema200) qty=0.50 sl=274.88 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 247.20 | 244.52 | 244.35 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 240.95 | 243.94 | 244.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 240.60 | 242.41 | 243.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 263.60 | 238.70 | 240.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 257.85 | 242.53 | 241.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 268.15 | 252.08 | 246.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 293.65 | 296.69 | 280.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 292.20 | 296.69 | 280.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 286.80 | 290.80 | 285.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 288.55 | 290.80 | 285.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 283.60 | 288.48 | 285.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 283.35 | 288.48 | 285.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 282.60 | 287.30 | 285.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 282.60 | 287.30 | 285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 279.20 | 283.89 | 284.28 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 287.20 | 284.42 | 284.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 293.90 | 286.31 | 284.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 300.00 | 300.34 | 296.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 303.85 | 300.34 | 296.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:00:00 | 302.50 | 300.77 | 297.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 300.05 | 300.41 | 297.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 297.85 | 300.41 | 297.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 297.95 | 299.62 | 297.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 297.50 | 299.62 | 297.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 297.70 | 299.23 | 297.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 297.70 | 299.23 | 297.71 | SL hit (close<ema400) qty=1.00 sl=297.71 alert=retest1 |

### Cycle 48 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 293.75 | 296.89 | 296.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 293.40 | 296.19 | 296.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 286.05 | 285.19 | 288.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 286.05 | 285.19 | 288.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 286.75 | 285.09 | 287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 287.90 | 285.09 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 286.80 | 285.58 | 287.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 287.20 | 285.58 | 287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 286.60 | 285.78 | 287.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 287.20 | 285.78 | 287.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 285.70 | 285.77 | 286.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 286.60 | 285.77 | 286.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 305.30 | 289.69 | 288.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 340.90 | 310.01 | 300.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 324.35 | 326.77 | 314.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:30:00 | 324.25 | 326.77 | 314.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 313.55 | 324.67 | 320.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 313.55 | 324.67 | 320.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 313.00 | 322.33 | 320.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:45:00 | 307.20 | 322.33 | 320.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 315.65 | 318.54 | 318.61 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 319.45 | 318.72 | 318.69 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 311.40 | 317.26 | 318.03 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 321.90 | 318.70 | 318.34 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 313.80 | 318.03 | 318.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 288.65 | 311.05 | 314.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 271.50 | 269.70 | 276.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 271.50 | 269.70 | 276.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 245.40 | 245.41 | 249.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 244.00 | 245.41 | 249.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 242.05 | 244.77 | 248.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 15:15:00 | 231.80 | 235.50 | 240.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 235.90 | 235.58 | 239.79 | SL hit (close>ema200) qty=0.50 sl=235.58 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 241.00 | 237.88 | 237.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 243.00 | 238.90 | 238.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 234.70 | 238.04 | 237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 232.40 | 236.91 | 237.27 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 238.23 | 236.06 | 235.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 229.68 | 235.10 | 235.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 225.80 | 233.24 | 234.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 231.50 | 230.53 | 232.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 240.45 | 230.53 | 232.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 247.10 | 233.84 | 234.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:00:00 | 247.10 | 233.84 | 234.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 247.22 | 236.52 | 235.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 256.00 | 246.06 | 241.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 250.64 | 251.28 | 246.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:30:00 | 248.16 | 251.28 | 246.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 246.82 | 251.05 | 248.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 246.82 | 251.05 | 248.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 247.90 | 250.42 | 248.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 245.49 | 250.42 | 248.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 248.33 | 250.14 | 248.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 248.33 | 250.14 | 248.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 248.13 | 249.74 | 248.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 247.71 | 249.74 | 248.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 248.48 | 249.26 | 248.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 261.83 | 249.26 | 248.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 14:15:00 | 288.01 | 270.89 | 261.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 273.37 | 278.89 | 279.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 270.50 | 274.90 | 276.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 273.00 | 270.59 | 273.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 269.98 | 270.47 | 273.49 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 298.15 | 278.63 | 276.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 311.35 | 297.45 | 288.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 302.30 | 304.10 | 296.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 308.26 | 304.10 | 296.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 304.20 | 311.28 | 305.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 304.20 | 311.28 | 305.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 305.11 | 310.05 | 305.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 303.95 | 310.05 | 305.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 303.73 | 308.78 | 305.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 302.50 | 308.78 | 305.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 303.21 | 306.35 | 305.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 303.21 | 306.35 | 305.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 304.30 | 305.94 | 305.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 301.42 | 305.94 | 305.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 306.66 | 306.08 | 305.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 303.33 | 306.08 | 305.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 305.08 | 305.88 | 305.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 305.08 | 305.88 | 305.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 302.10 | 305.13 | 304.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 302.10 | 305.13 | 304.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 302.47 | 304.59 | 304.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 299.34 | 302.79 | 303.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 303.19 | 299.03 | 300.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 315.34 | 302.29 | 302.09 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 298.05 | 306.93 | 307.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 296.60 | 304.87 | 306.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 305.50 | 303.74 | 305.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 306.00 | 304.19 | 305.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 294.60 | 304.19 | 305.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 279.87 | 291.89 | 294.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 282.00 | 281.15 | 286.39 | SL hit (close>ema200) qty=0.50 sl=281.15 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 288.55 | 287.22 | 287.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 293.00 | 290.29 | 289.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 280.15 | 288.85 | 288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 278.35 | 286.75 | 287.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 277.50 | 283.84 | 286.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 271.10 | 267.87 | 273.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 271.10 | 267.87 | 273.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 275.20 | 269.34 | 273.69 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 284.75 | 277.51 | 276.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 288.15 | 281.58 | 278.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 273.00 | 280.09 | 280.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 284.25 | 281.35 | 281.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 285.30 | 282.51 | 281.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 274.25 | 282.02 | 281.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 275.40 | 280.69 | 281.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 271.30 | 277.85 | 279.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 274.25 | 274.22 | 277.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 287.55 | 279.65 | 278.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 287.55 | 279.65 | 278.65 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 273.05 | 279.38 | 279.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 271.50 | 274.97 | 277.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 273.80 | 271.17 | 270.99 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 266.45 | 270.81 | 270.90 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 272.10 | 270.85 | 270.73 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 268.50 | 270.38 | 270.52 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 272.45 | 270.69 | 270.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 274.25 | 272.24 | 271.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 268.70 | 270.95 | 271.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 268.20 | 270.40 | 270.87 | Break + close below crossover candle low |

### Cycle 79 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 280.00 | 271.73 | 271.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 290.50 | 283.57 | 280.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 319.30 | 320.00 | 313.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:45:00 | 318.15 | 320.00 | 313.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 305.45 | 324.67 | 322.40 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 305.05 | 320.75 | 320.82 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 322.05 | 317.99 | 317.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 324.10 | 320.50 | 319.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 338.75 | 341.19 | 334.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 338.75 | 341.19 | 334.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 337.95 | 339.37 | 335.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 332.30 | 339.37 | 335.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 329.00 | 337.30 | 335.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 329.00 | 337.30 | 335.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 328.30 | 335.50 | 334.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 327.90 | 335.50 | 334.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 326.80 | 333.76 | 333.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 324.30 | 329.93 | 331.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 329.90 | 327.14 | 329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 329.80 | 327.67 | 329.69 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 332.35 | 330.76 | 330.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 332.70 | 331.15 | 330.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 329.75 | 331.66 | 331.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 329.00 | 331.13 | 330.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 329.30 | 331.13 | 330.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 323.40 | 329.58 | 330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 318.45 | 325.15 | 327.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 321.90 | 317.57 | 321.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 320.00 | 318.05 | 321.56 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 327.65 | 324.07 | 323.66 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 09:30:00 | 525.75 | 2025-05-28 14:15:00 | 511.35 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-05-28 10:15:00 | 522.90 | 2025-05-28 14:15:00 | 511.35 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-02 10:30:00 | 535.00 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-03 09:15:00 | 536.60 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-06-04 09:15:00 | 536.10 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-04 11:30:00 | 539.10 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-06-10 12:15:00 | 517.95 | 2025-06-13 09:15:00 | 492.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 09:45:00 | 518.20 | 2025-06-13 09:15:00 | 492.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 12:15:00 | 517.95 | 2025-06-16 14:15:00 | 494.95 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-06-11 09:45:00 | 518.20 | 2025-06-16 14:15:00 | 494.95 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest2 | 2025-06-30 09:15:00 | 499.15 | 2025-06-30 15:15:00 | 488.80 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-04 11:15:00 | 469.05 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-07-07 09:30:00 | 468.30 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-07-07 10:15:00 | 469.20 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-07-07 10:45:00 | 469.10 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-07-10 12:15:00 | 514.30 | 2025-07-11 09:15:00 | 501.05 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-07-15 15:00:00 | 513.00 | 2025-07-21 09:15:00 | 511.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-16 10:15:00 | 517.50 | 2025-07-21 09:15:00 | 511.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest1 | 2025-08-01 11:30:00 | 519.50 | 2025-08-04 12:15:00 | 545.48 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-08-01 11:30:00 | 519.50 | 2025-08-05 09:15:00 | 571.45 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-04 12:45:00 | 532.25 | 2025-08-06 09:15:00 | 508.00 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-08-08 10:45:00 | 499.75 | 2025-08-11 10:15:00 | 529.70 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2025-08-18 12:00:00 | 505.10 | 2025-08-25 11:15:00 | 510.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-29 12:15:00 | 494.85 | 2025-09-01 10:15:00 | 501.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-01 10:00:00 | 494.70 | 2025-09-01 10:15:00 | 501.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-01 10:30:00 | 495.40 | 2025-09-01 11:15:00 | 504.15 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-08 09:15:00 | 499.20 | 2025-09-10 09:15:00 | 505.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-19 09:15:00 | 526.50 | 2025-09-24 11:15:00 | 523.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-10 11:15:00 | 491.20 | 2025-10-10 13:15:00 | 492.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-10 13:15:00 | 491.30 | 2025-10-10 13:15:00 | 492.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-10-14 09:45:00 | 487.25 | 2025-10-28 13:15:00 | 462.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 486.45 | 2025-10-28 13:15:00 | 462.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-14 09:45:00 | 487.25 | 2025-10-29 09:15:00 | 466.65 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-10-16 09:30:00 | 486.45 | 2025-10-29 09:15:00 | 466.65 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-11-25 14:15:00 | 299.60 | 2025-11-27 11:15:00 | 284.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:15:00 | 299.60 | 2025-12-01 09:15:00 | 275.20 | STOP_HIT | 0.50 | 8.14% |
| BUY | retest1 | 2025-12-24 09:15:00 | 303.85 | 2025-12-24 14:15:00 | 297.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-12-24 10:00:00 | 302.50 | 2025-12-24 14:15:00 | 297.70 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-22 10:15:00 | 244.00 | 2026-01-23 15:15:00 | 231.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 244.00 | 2026-01-27 09:15:00 | 235.90 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2026-01-22 10:45:00 | 242.05 | 2026-01-27 09:15:00 | 229.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 242.05 | 2026-01-27 09:15:00 | 235.90 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest2 | 2026-02-09 09:15:00 | 261.83 | 2026-02-09 14:15:00 | 288.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 294.60 | 2026-03-09 09:15:00 | 279.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 294.60 | 2026-03-10 09:15:00 | 282.00 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2026-03-24 10:15:00 | 274.25 | 2026-03-25 09:15:00 | 287.55 | STOP_HIT | 1.00 | -4.85% |
