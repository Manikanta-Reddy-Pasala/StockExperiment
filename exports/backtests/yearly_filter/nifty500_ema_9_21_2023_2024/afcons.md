# Afcons Infrastructure Ltd. (AFCONS)

## Backtest Summary

- **Window:** 2024-11-04 09:15:00 → 2026-05-11 15:15:00 (2613 bars)
- **Last close:** 326.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 113 |
| ALERT1 | 78 |
| ALERT2 | 76 |
| ALERT2_SKIP | 48 |
| ALERT3 | 191 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 83 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 58
- **Target hits / Stop hits / Partials:** 7 / 78 / 7
- **Avg / median % per leg:** 0.64% / -0.85%
- **Sum % (uncompounded):** 59.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 15 | 31.9% | 4 | 43 | 0 | 0.40% | 18.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 47 | 15 | 31.9% | 4 | 43 | 0 | 0.40% | 18.7% |
| SELL (all) | 45 | 19 | 42.2% | 3 | 35 | 7 | 0.90% | 40.5% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.46% | 7.4% |
| SELL @ 3rd Alert (retest2) | 42 | 17 | 40.5% | 3 | 33 | 6 | 0.79% | 33.2% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.46% | 7.4% |
| retest2 (combined) | 89 | 32 | 36.0% | 7 | 76 | 6 | 0.58% | 51.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 476.40 | 479.40 | 479.75 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 489.80 | 480.88 | 480.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 12:15:00 | 512.00 | 488.95 | 484.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 09:15:00 | 488.60 | 489.65 | 486.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 488.60 | 489.65 | 486.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 488.60 | 489.65 | 486.20 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 481.85 | 486.77 | 486.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 475.45 | 484.51 | 485.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 477.00 | 471.19 | 476.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 12:15:00 | 477.00 | 471.19 | 476.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 477.00 | 471.19 | 476.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:00:00 | 477.00 | 471.19 | 476.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 475.30 | 472.01 | 476.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 14:00:00 | 475.30 | 472.01 | 476.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 468.55 | 471.32 | 475.49 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 492.60 | 473.70 | 473.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 499.10 | 482.00 | 477.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 480.55 | 485.14 | 480.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 480.55 | 485.14 | 480.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 480.55 | 485.14 | 480.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 480.55 | 485.14 | 480.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 475.15 | 483.15 | 479.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 487.25 | 483.15 | 479.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:45:00 | 481.90 | 483.85 | 480.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 12:00:00 | 481.30 | 485.07 | 483.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-25 14:15:00 | 530.09 | 506.98 | 497.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 14:15:00 | 495.30 | 508.00 | 509.67 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 512.10 | 508.62 | 508.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 11:15:00 | 514.75 | 509.92 | 509.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 14:15:00 | 504.20 | 508.91 | 508.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 504.20 | 508.91 | 508.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 504.20 | 508.91 | 508.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 504.20 | 508.91 | 508.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 502.45 | 507.61 | 508.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 496.25 | 505.34 | 507.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 12:15:00 | 493.40 | 493.20 | 496.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:00:00 | 493.40 | 493.20 | 496.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 498.10 | 494.45 | 496.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 498.10 | 494.45 | 496.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 496.90 | 494.94 | 496.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 503.80 | 494.94 | 496.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 500.85 | 496.12 | 496.65 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 501.55 | 497.21 | 497.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 11:15:00 | 507.65 | 499.30 | 498.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 536.20 | 537.30 | 525.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 10:00:00 | 536.20 | 537.30 | 525.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 525.40 | 532.18 | 527.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 525.30 | 532.18 | 527.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 525.95 | 530.94 | 527.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 530.70 | 530.94 | 527.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:45:00 | 526.70 | 530.34 | 527.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 530.00 | 530.19 | 530.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 528.05 | 529.76 | 529.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 528.05 | 529.76 | 529.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 524.20 | 528.37 | 529.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 528.70 | 528.44 | 529.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 12:15:00 | 528.70 | 528.44 | 529.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 528.70 | 528.44 | 529.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:00:00 | 528.70 | 528.44 | 529.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 524.10 | 523.86 | 526.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 524.10 | 523.86 | 526.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 525.60 | 524.21 | 526.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 526.45 | 524.21 | 526.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 522.30 | 523.82 | 525.70 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 545.70 | 528.05 | 527.28 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 531.90 | 536.80 | 536.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 14:15:00 | 528.55 | 535.15 | 536.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 519.75 | 519.19 | 524.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 519.75 | 519.19 | 524.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 519.75 | 519.19 | 524.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 521.00 | 519.19 | 524.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 527.85 | 521.67 | 525.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 527.85 | 521.67 | 525.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 526.15 | 522.56 | 525.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 521.50 | 522.44 | 524.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 532.35 | 523.74 | 524.88 | SL hit (close>static) qty=1.00 sl=528.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 534.85 | 525.96 | 525.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 537.20 | 532.31 | 529.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 532.40 | 534.15 | 531.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 532.40 | 534.15 | 531.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 532.40 | 534.15 | 531.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:30:00 | 529.45 | 534.15 | 531.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 530.65 | 533.42 | 531.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 530.65 | 533.42 | 531.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 529.55 | 532.64 | 531.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 528.65 | 532.64 | 531.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 528.35 | 530.70 | 530.79 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 533.00 | 530.82 | 530.81 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 524.75 | 529.84 | 530.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 521.50 | 528.17 | 529.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 535.95 | 523.79 | 525.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 535.95 | 523.79 | 525.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 535.95 | 523.79 | 525.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 535.95 | 523.79 | 525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 545.20 | 528.07 | 527.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 15:15:00 | 546.40 | 531.74 | 528.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 538.15 | 538.22 | 534.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:15:00 | 537.90 | 538.22 | 534.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 534.75 | 537.26 | 534.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 534.80 | 537.26 | 534.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 537.20 | 537.25 | 535.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:15:00 | 542.25 | 536.31 | 535.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 540.05 | 544.14 | 541.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 533.25 | 540.67 | 540.23 | SL hit (close<static) qty=1.00 sl=534.40 alert=retest2 |

### Cycle 17 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 525.75 | 537.68 | 538.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 523.65 | 528.28 | 531.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 526.80 | 524.64 | 528.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 526.80 | 524.64 | 528.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 526.80 | 524.64 | 528.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 527.95 | 524.64 | 528.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 521.00 | 523.74 | 527.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 518.00 | 522.11 | 526.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 492.10 | 503.03 | 511.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 466.20 | 485.15 | 498.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 18 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 509.70 | 498.61 | 498.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 511.45 | 505.08 | 501.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 519.50 | 520.50 | 514.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:45:00 | 519.95 | 520.50 | 514.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 518.55 | 519.96 | 516.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:45:00 | 523.80 | 520.81 | 517.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:30:00 | 524.00 | 521.42 | 518.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:00:00 | 523.90 | 521.42 | 518.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 525.00 | 521.79 | 518.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 518.70 | 521.69 | 519.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 518.70 | 521.69 | 519.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 510.60 | 519.47 | 518.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 510.60 | 519.47 | 518.60 | SL hit (close<static) qty=1.00 sl=516.10 alert=retest2 |

### Cycle 19 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 509.35 | 517.44 | 517.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 507.85 | 515.53 | 516.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 508.65 | 502.40 | 506.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 508.65 | 502.40 | 506.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 508.65 | 502.40 | 506.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 508.65 | 502.40 | 506.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 503.05 | 502.53 | 506.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 499.65 | 501.49 | 505.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 474.67 | 483.57 | 492.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 10:15:00 | 449.69 | 476.81 | 488.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 20 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 467.00 | 459.93 | 459.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 485.15 | 470.03 | 465.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 468.50 | 474.52 | 468.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 468.50 | 474.52 | 468.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 468.50 | 474.52 | 468.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 468.50 | 474.52 | 468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 475.00 | 474.61 | 469.43 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 444.45 | 463.41 | 465.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 442.45 | 459.22 | 463.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 446.05 | 435.71 | 443.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 446.05 | 435.71 | 443.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 446.05 | 435.71 | 443.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 451.45 | 435.71 | 443.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 445.80 | 437.73 | 443.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 447.40 | 437.73 | 443.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 464.25 | 447.36 | 446.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 13:15:00 | 466.70 | 460.82 | 456.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 11:15:00 | 460.30 | 462.99 | 459.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 11:15:00 | 460.30 | 462.99 | 459.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 460.30 | 462.99 | 459.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 460.30 | 462.99 | 459.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 461.05 | 462.60 | 459.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:30:00 | 462.55 | 462.60 | 459.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 455.00 | 461.08 | 459.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:00:00 | 455.00 | 461.08 | 459.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 457.55 | 460.38 | 459.23 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 449.40 | 458.09 | 458.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 442.75 | 453.29 | 456.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 450.60 | 448.49 | 452.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 14:15:00 | 450.60 | 448.49 | 452.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 450.60 | 448.49 | 452.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 450.60 | 448.49 | 452.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 430.50 | 445.44 | 450.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 428.95 | 441.70 | 446.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:45:00 | 429.20 | 426.56 | 434.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 429.00 | 426.16 | 433.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:45:00 | 428.40 | 426.70 | 431.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 430.75 | 427.51 | 431.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 430.75 | 427.51 | 431.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 430.50 | 428.75 | 431.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 434.70 | 433.06 | 432.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 434.70 | 433.06 | 432.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 449.35 | 436.32 | 434.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 447.85 | 451.12 | 446.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 447.85 | 451.12 | 446.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 451.65 | 451.22 | 447.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 457.00 | 450.24 | 449.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 446.85 | 449.08 | 449.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 446.85 | 449.08 | 449.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 14:15:00 | 444.55 | 447.14 | 448.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 420.50 | 411.34 | 419.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 420.50 | 411.34 | 419.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 420.50 | 411.34 | 419.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 421.05 | 411.34 | 419.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 421.05 | 413.28 | 419.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 421.05 | 413.28 | 419.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 416.00 | 413.82 | 419.15 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 429.95 | 422.16 | 421.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 439.10 | 425.55 | 422.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 13:15:00 | 455.65 | 457.77 | 451.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 455.65 | 457.77 | 451.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 461.80 | 460.31 | 457.03 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 448.70 | 455.62 | 456.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 11:15:00 | 444.65 | 451.23 | 453.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 451.25 | 449.17 | 451.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 451.25 | 449.17 | 451.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 451.25 | 449.17 | 451.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 451.25 | 449.17 | 451.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 451.30 | 449.59 | 451.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 448.80 | 449.59 | 451.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 450.95 | 449.86 | 451.83 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 455.85 | 452.15 | 451.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 462.80 | 456.26 | 454.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 484.45 | 486.33 | 479.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 484.00 | 486.33 | 479.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 478.30 | 484.12 | 479.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 476.65 | 484.12 | 479.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 477.55 | 482.81 | 479.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 477.55 | 482.81 | 479.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 476.65 | 481.13 | 479.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 476.65 | 481.13 | 479.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 477.00 | 480.30 | 479.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 479.15 | 480.30 | 479.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 479.25 | 480.09 | 479.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 480.05 | 480.41 | 479.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 475.30 | 479.21 | 479.04 | SL hit (close<static) qty=1.00 sl=476.20 alert=retest2 |

### Cycle 29 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 472.30 | 477.83 | 478.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 468.80 | 476.02 | 477.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 472.70 | 470.02 | 473.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 472.70 | 470.02 | 473.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 477.85 | 471.59 | 473.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 475.90 | 471.59 | 473.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 477.20 | 472.71 | 473.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 486.50 | 472.71 | 473.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 483.95 | 474.96 | 474.88 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 474.55 | 477.48 | 477.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 471.80 | 476.35 | 477.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 473.30 | 473.26 | 475.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 473.30 | 473.26 | 475.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 473.30 | 473.26 | 475.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 474.70 | 473.26 | 475.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 476.00 | 473.81 | 475.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:30:00 | 474.50 | 473.81 | 475.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 473.50 | 473.75 | 475.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 473.70 | 473.75 | 475.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 470.85 | 472.42 | 474.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:00:00 | 462.25 | 469.46 | 471.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 416.03 | 450.37 | 460.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 444.75 | 432.23 | 431.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 446.85 | 435.15 | 432.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 440.55 | 443.36 | 440.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 440.55 | 443.36 | 440.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 440.55 | 443.36 | 440.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:00:00 | 446.95 | 444.10 | 441.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 446.30 | 444.39 | 442.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 446.00 | 444.39 | 442.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:30:00 | 446.20 | 445.08 | 443.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 444.00 | 444.86 | 443.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 443.85 | 444.86 | 443.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 444.40 | 444.77 | 443.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:30:00 | 444.40 | 444.77 | 443.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 445.50 | 444.92 | 443.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 448.30 | 444.92 | 443.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 12:15:00 | 443.00 | 445.38 | 444.41 | SL hit (close<static) qty=1.00 sl=443.60 alert=retest2 |

### Cycle 33 — SELL (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 15:15:00 | 442.40 | 443.89 | 443.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 09:15:00 | 438.00 | 442.71 | 443.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 425.30 | 423.64 | 426.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 425.30 | 423.64 | 426.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 425.30 | 423.64 | 426.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:45:00 | 421.80 | 425.26 | 426.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:45:00 | 421.65 | 424.21 | 425.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 420.60 | 421.60 | 423.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:30:00 | 420.95 | 420.56 | 422.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 418.90 | 419.98 | 421.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:45:00 | 412.80 | 415.89 | 417.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 426.60 | 418.28 | 418.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 426.60 | 418.28 | 418.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 428.05 | 421.50 | 419.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 422.75 | 425.12 | 422.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 422.75 | 425.12 | 422.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 422.75 | 425.12 | 422.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 422.75 | 425.12 | 422.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 422.20 | 424.54 | 422.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:15:00 | 420.55 | 424.54 | 422.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 418.85 | 423.40 | 422.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 419.60 | 423.40 | 422.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 414.90 | 420.83 | 421.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 412.95 | 419.25 | 420.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 417.60 | 417.28 | 418.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 417.60 | 417.28 | 418.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 418.80 | 417.58 | 418.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:45:00 | 418.60 | 417.58 | 418.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 420.00 | 418.07 | 419.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 432.25 | 418.07 | 419.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 432.55 | 420.96 | 420.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 451.20 | 433.30 | 427.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 450.00 | 450.82 | 443.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 448.75 | 450.82 | 443.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 455.95 | 453.98 | 450.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 464.15 | 455.19 | 451.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 455.45 | 463.41 | 463.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 455.45 | 463.41 | 463.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 12:15:00 | 451.50 | 461.03 | 462.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 449.90 | 449.46 | 453.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 446.70 | 448.68 | 452.91 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 437.40 | 436.66 | 439.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 434.50 | 436.66 | 439.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:15:00 | 424.36 | 428.69 | 432.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 425.45 | 424.73 | 427.70 | SL hit (close>ema200) qty=0.50 sl=424.73 alert=retest1 |

### Cycle 38 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 428.55 | 424.31 | 423.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 448.20 | 433.73 | 429.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 445.15 | 445.97 | 440.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:00:00 | 445.15 | 445.97 | 440.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 445.00 | 448.11 | 446.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 445.00 | 448.11 | 446.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 446.00 | 447.68 | 446.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:30:00 | 447.00 | 447.68 | 446.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 445.00 | 447.15 | 445.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 439.25 | 447.15 | 445.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 442.25 | 446.17 | 445.58 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 438.05 | 443.89 | 444.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 435.35 | 442.18 | 443.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 439.75 | 438.99 | 441.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:45:00 | 439.75 | 438.99 | 441.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 442.05 | 439.60 | 441.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 442.05 | 439.60 | 441.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 443.15 | 440.31 | 441.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 443.15 | 440.31 | 441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 443.10 | 440.87 | 441.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 441.50 | 440.87 | 441.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 445.00 | 442.21 | 442.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 445.00 | 442.21 | 442.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 447.40 | 443.25 | 442.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 442.00 | 443.93 | 443.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 442.00 | 443.93 | 443.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 442.00 | 443.93 | 443.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 442.00 | 443.93 | 443.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 444.55 | 444.06 | 443.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 442.25 | 444.06 | 443.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 439.70 | 443.18 | 443.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 439.70 | 443.18 | 443.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 436.85 | 441.92 | 442.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 434.15 | 439.97 | 441.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 438.70 | 429.41 | 432.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 438.70 | 429.41 | 432.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 438.70 | 429.41 | 432.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 438.70 | 429.41 | 432.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 455.75 | 434.68 | 434.28 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 432.60 | 433.77 | 433.90 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 435.40 | 434.10 | 434.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 438.75 | 435.53 | 434.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 437.70 | 438.96 | 437.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 437.70 | 438.96 | 437.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 437.85 | 438.74 | 437.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 440.05 | 438.74 | 437.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:30:00 | 439.10 | 438.87 | 438.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 440.90 | 438.66 | 438.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 435.65 | 438.37 | 438.06 | SL hit (close<static) qty=1.00 sl=436.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 434.20 | 437.53 | 437.71 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 439.05 | 437.67 | 437.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 444.45 | 439.03 | 438.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 438.00 | 441.14 | 440.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 438.00 | 441.14 | 440.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 438.00 | 441.14 | 440.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 438.00 | 441.14 | 440.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 437.10 | 440.34 | 440.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 437.10 | 440.34 | 440.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 435.60 | 439.39 | 439.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 434.00 | 437.87 | 438.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 430.00 | 429.97 | 432.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 430.00 | 429.97 | 432.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 432.40 | 430.49 | 432.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 428.75 | 431.71 | 432.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 428.05 | 429.75 | 430.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 428.70 | 425.74 | 425.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 426.75 | 426.14 | 426.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 426.75 | 426.14 | 426.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 428.55 | 426.63 | 426.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 424.60 | 426.43 | 426.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 424.60 | 426.43 | 426.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 424.60 | 426.43 | 426.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 424.00 | 426.43 | 426.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 420.40 | 425.23 | 425.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 12:15:00 | 418.50 | 419.93 | 421.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 421.55 | 418.91 | 420.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 421.55 | 418.91 | 420.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 421.55 | 418.91 | 420.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 420.40 | 420.67 | 420.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 425.30 | 420.16 | 420.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 425.30 | 420.16 | 420.03 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 422.95 | 423.68 | 423.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 421.30 | 423.21 | 423.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 401.65 | 398.38 | 403.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 401.65 | 398.38 | 403.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 401.65 | 398.38 | 403.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 396.10 | 398.29 | 401.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 406.20 | 401.70 | 401.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 406.20 | 401.70 | 401.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 408.00 | 405.83 | 404.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 14:15:00 | 407.30 | 407.79 | 406.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:45:00 | 407.45 | 407.79 | 406.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 404.05 | 407.00 | 406.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 403.90 | 407.00 | 406.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 403.65 | 406.33 | 405.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 403.65 | 406.33 | 405.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 404.75 | 405.37 | 405.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 403.60 | 405.02 | 405.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 407.50 | 405.19 | 405.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 407.50 | 405.19 | 405.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 407.50 | 405.19 | 405.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 407.10 | 405.19 | 405.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 406.55 | 405.46 | 405.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 410.75 | 406.88 | 406.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 405.90 | 409.30 | 407.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 405.90 | 409.30 | 407.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 405.90 | 409.30 | 407.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 405.90 | 409.30 | 407.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 405.90 | 408.62 | 407.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 420.15 | 408.62 | 407.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 428.35 | 430.34 | 430.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 428.35 | 430.34 | 430.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 426.35 | 429.09 | 429.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 431.25 | 428.64 | 429.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 431.25 | 428.64 | 429.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 431.25 | 428.64 | 429.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 431.25 | 428.64 | 429.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 15:15:00 | 443.00 | 431.51 | 430.74 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 427.05 | 429.79 | 430.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 421.85 | 428.20 | 429.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 422.25 | 420.31 | 423.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 422.25 | 420.31 | 423.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 422.25 | 420.31 | 423.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 422.60 | 420.31 | 423.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 425.00 | 421.25 | 423.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 425.00 | 421.25 | 423.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 424.30 | 421.86 | 423.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 425.50 | 421.86 | 423.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 425.45 | 422.71 | 423.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 425.45 | 422.71 | 423.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 426.65 | 423.50 | 423.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 425.85 | 423.50 | 423.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 428.00 | 424.40 | 424.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 429.70 | 426.19 | 425.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 425.35 | 426.87 | 425.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 425.35 | 426.87 | 425.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 425.35 | 426.87 | 425.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 425.35 | 426.87 | 425.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 426.00 | 426.70 | 425.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 425.50 | 426.70 | 425.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 426.50 | 426.66 | 425.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 428.25 | 426.66 | 425.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 438.00 | 428.93 | 426.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 439.65 | 432.45 | 428.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 440.10 | 435.51 | 432.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 440.50 | 439.23 | 435.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 443.10 | 439.64 | 436.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 436.90 | 438.94 | 436.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 436.90 | 438.94 | 436.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 437.00 | 438.55 | 436.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 436.75 | 438.55 | 436.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 438.15 | 438.47 | 436.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 440.65 | 438.97 | 437.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 439.50 | 438.94 | 437.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:45:00 | 439.60 | 438.95 | 437.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 439.75 | 438.84 | 438.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 439.40 | 438.95 | 438.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 442.10 | 438.63 | 438.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 450.40 | 453.17 | 453.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 450.40 | 453.17 | 453.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 448.45 | 451.21 | 452.30 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 461.70 | 453.31 | 453.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 468.25 | 456.30 | 454.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 455.95 | 457.96 | 456.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 455.95 | 457.96 | 456.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 455.95 | 457.96 | 456.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 455.05 | 457.96 | 456.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 453.05 | 456.98 | 456.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 453.05 | 456.98 | 456.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 454.00 | 455.87 | 455.88 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 455.95 | 455.75 | 455.75 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 453.40 | 455.28 | 455.53 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 457.80 | 455.83 | 455.74 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 453.15 | 455.37 | 455.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 452.45 | 454.78 | 455.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 451.25 | 443.00 | 445.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 451.25 | 443.00 | 445.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 451.25 | 443.00 | 445.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 451.25 | 443.00 | 445.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 448.50 | 444.10 | 446.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 447.20 | 444.77 | 446.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 447.15 | 445.61 | 446.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 445.00 | 446.15 | 446.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 445.20 | 446.02 | 446.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 447.00 | 446.22 | 446.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 455.00 | 448.10 | 447.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 455.00 | 448.10 | 447.22 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 448.00 | 449.66 | 449.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 446.50 | 449.03 | 449.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 448.35 | 448.19 | 448.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 448.35 | 448.19 | 448.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 448.35 | 448.19 | 448.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 448.35 | 448.19 | 448.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 448.50 | 448.26 | 448.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 450.00 | 448.26 | 448.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 450.05 | 448.61 | 449.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 451.15 | 448.61 | 449.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 449.55 | 448.80 | 449.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 449.55 | 448.80 | 449.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 450.05 | 449.05 | 449.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 450.05 | 449.05 | 449.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 450.85 | 449.41 | 449.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 452.40 | 450.21 | 449.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 464.15 | 466.50 | 462.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 464.15 | 466.50 | 462.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 464.15 | 466.50 | 462.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 465.30 | 466.50 | 462.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 463.00 | 465.80 | 462.93 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 454.90 | 461.08 | 461.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 11:15:00 | 451.60 | 454.96 | 456.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 446.65 | 446.64 | 450.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 13:00:00 | 446.65 | 446.64 | 450.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 447.05 | 446.96 | 448.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 443.20 | 445.20 | 446.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 443.30 | 444.73 | 446.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 443.30 | 444.27 | 445.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 443.25 | 443.74 | 445.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 443.85 | 442.32 | 443.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 444.45 | 442.32 | 443.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 446.20 | 443.10 | 444.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 446.20 | 443.10 | 444.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 446.35 | 443.75 | 444.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 447.00 | 444.97 | 444.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 447.00 | 444.97 | 444.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 451.90 | 447.52 | 446.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 451.65 | 452.10 | 450.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 451.65 | 452.10 | 450.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 449.90 | 451.60 | 450.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 449.90 | 451.60 | 450.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 450.90 | 451.46 | 450.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 453.75 | 451.87 | 450.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 454.10 | 452.33 | 450.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 448.60 | 451.44 | 450.88 | SL hit (close<static) qty=1.00 sl=450.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 445.95 | 450.01 | 450.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 440.50 | 445.17 | 447.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 442.35 | 442.35 | 444.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 442.30 | 442.35 | 444.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 439.20 | 441.75 | 443.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 438.25 | 440.71 | 442.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 13:15:00 | 416.34 | 424.21 | 432.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 421.40 | 421.11 | 427.21 | SL hit (close>ema200) qty=0.50 sl=421.11 alert=retest2 |

### Cycle 72 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 402.05 | 397.69 | 397.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 403.00 | 400.19 | 398.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 408.50 | 409.32 | 406.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 408.05 | 409.32 | 406.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 406.05 | 408.43 | 406.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 406.05 | 408.43 | 406.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 403.15 | 407.38 | 406.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 402.70 | 407.38 | 406.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 403.50 | 406.60 | 406.03 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 402.85 | 405.32 | 405.54 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 410.30 | 406.22 | 405.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 412.00 | 407.87 | 406.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 405.50 | 410.48 | 409.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 405.50 | 410.48 | 409.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 405.50 | 410.48 | 409.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 405.20 | 410.48 | 409.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 402.65 | 408.91 | 408.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 402.65 | 408.91 | 408.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 403.30 | 407.79 | 408.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 398.35 | 400.68 | 403.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 398.50 | 398.23 | 400.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 15:00:00 | 398.50 | 398.23 | 400.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 396.65 | 398.12 | 400.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 395.70 | 397.70 | 399.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 401.00 | 394.47 | 395.44 | SL hit (close>static) qty=1.00 sl=400.75 alert=retest2 |

### Cycle 76 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 399.25 | 396.15 | 396.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 401.60 | 397.87 | 397.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 398.95 | 399.51 | 398.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 13:15:00 | 398.95 | 399.51 | 398.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 398.95 | 399.51 | 398.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 398.95 | 399.51 | 398.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 400.00 | 399.61 | 398.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 397.70 | 399.61 | 398.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 398.55 | 399.40 | 398.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 398.25 | 399.40 | 398.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 398.25 | 399.17 | 398.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 397.65 | 399.17 | 398.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 397.50 | 398.83 | 398.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 397.50 | 398.83 | 398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 398.50 | 398.77 | 398.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 399.10 | 398.70 | 398.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 399.30 | 398.93 | 398.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 395.90 | 398.46 | 398.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 395.90 | 398.46 | 398.50 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 399.70 | 398.71 | 398.61 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 395.75 | 398.25 | 398.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 395.10 | 397.62 | 398.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 380.85 | 380.13 | 384.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 380.85 | 380.13 | 384.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 383.50 | 380.92 | 383.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:45:00 | 381.00 | 381.54 | 383.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 385.85 | 383.60 | 383.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 385.85 | 383.60 | 383.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 11:15:00 | 390.00 | 384.88 | 383.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 386.30 | 388.79 | 387.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 386.30 | 388.79 | 387.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 386.30 | 388.79 | 387.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 386.30 | 388.79 | 387.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 382.25 | 387.48 | 387.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 382.25 | 387.48 | 387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 385.10 | 387.01 | 386.84 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 385.20 | 386.65 | 386.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 383.80 | 385.89 | 386.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 386.00 | 385.45 | 386.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 386.00 | 385.45 | 386.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 386.00 | 385.45 | 386.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 387.20 | 385.45 | 386.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 386.95 | 385.75 | 386.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 387.20 | 385.75 | 386.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 390.55 | 386.71 | 386.50 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 386.25 | 387.48 | 387.51 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 392.25 | 388.07 | 387.71 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 383.70 | 387.65 | 388.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 380.50 | 383.31 | 385.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 378.75 | 378.35 | 380.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:45:00 | 378.95 | 378.35 | 380.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 361.50 | 359.58 | 362.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 363.15 | 359.58 | 362.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 362.75 | 360.22 | 362.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 362.40 | 360.22 | 362.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 362.25 | 360.62 | 362.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 360.75 | 360.62 | 362.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 358.10 | 360.95 | 362.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 342.71 | 347.46 | 351.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 350.15 | 347.76 | 350.92 | SL hit (close>ema200) qty=0.50 sl=347.76 alert=retest2 |

### Cycle 86 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 340.65 | 333.82 | 333.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 343.50 | 335.75 | 334.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 334.70 | 336.19 | 335.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 334.70 | 336.19 | 335.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 334.70 | 336.19 | 335.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 334.70 | 336.19 | 335.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 333.15 | 335.58 | 334.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 329.00 | 335.58 | 334.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 329.85 | 334.44 | 334.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 326.40 | 332.83 | 333.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 334.40 | 331.72 | 332.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 334.40 | 331.72 | 332.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 334.40 | 331.72 | 332.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 334.40 | 331.72 | 332.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 333.80 | 332.14 | 332.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 339.20 | 332.14 | 332.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 337.10 | 333.82 | 333.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 338.90 | 335.32 | 334.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 341.75 | 342.31 | 339.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 341.75 | 342.31 | 339.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 340.00 | 341.85 | 339.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 340.00 | 341.85 | 339.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 340.10 | 341.50 | 339.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 340.00 | 341.50 | 339.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 339.50 | 341.10 | 339.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 339.50 | 341.10 | 339.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 338.45 | 340.57 | 339.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:30:00 | 338.00 | 340.57 | 339.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 338.05 | 340.07 | 339.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 333.90 | 340.07 | 339.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 333.75 | 338.80 | 338.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 330.25 | 336.08 | 337.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 334.95 | 333.80 | 335.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 334.95 | 333.80 | 335.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 334.95 | 333.80 | 335.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 336.70 | 333.80 | 335.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 337.80 | 334.60 | 335.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 337.80 | 334.60 | 335.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 338.40 | 335.36 | 336.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 338.40 | 335.36 | 336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 340.30 | 337.30 | 336.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 341.95 | 338.23 | 337.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 345.00 | 345.14 | 342.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 329.80 | 345.14 | 342.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 334.70 | 343.05 | 341.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 338.40 | 341.44 | 340.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 13:15:00 | 338.40 | 340.13 | 340.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 338.40 | 340.13 | 340.31 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 344.50 | 341.00 | 340.69 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 337.95 | 340.46 | 340.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 330.70 | 336.69 | 338.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 320.10 | 319.35 | 321.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 320.10 | 319.35 | 321.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 317.00 | 319.02 | 320.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 314.50 | 319.02 | 320.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:15:00 | 298.77 | 307.03 | 310.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 308.75 | 305.05 | 308.23 | SL hit (close>ema200) qty=0.50 sl=305.05 alert=retest2 |

### Cycle 94 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 288.25 | 284.25 | 283.94 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 272.50 | 282.24 | 283.31 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 284.45 | 280.61 | 280.25 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 273.50 | 279.66 | 280.16 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 282.35 | 280.65 | 280.51 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 279.20 | 280.40 | 280.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 277.80 | 279.64 | 280.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 279.50 | 279.20 | 279.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:15:00 | 274.00 | 279.20 | 279.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 279.00 | 278.38 | 279.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 279.00 | 278.38 | 279.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 278.55 | 278.42 | 279.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:30:00 | 279.00 | 278.42 | 279.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 278.65 | 278.46 | 278.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 280.05 | 278.46 | 278.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 278.50 | 278.47 | 278.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 277.50 | 278.47 | 278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 277.25 | 278.22 | 278.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 280.45 | 278.22 | 278.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 280.50 | 278.68 | 278.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 280.50 | 278.68 | 278.92 | SL hit (close>ema400) qty=1.00 sl=278.92 alert=retest1 |

### Cycle 100 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 280.70 | 279.08 | 279.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 282.20 | 279.71 | 279.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 284.15 | 285.97 | 283.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 284.15 | 285.97 | 283.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 284.15 | 285.97 | 283.74 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 278.00 | 282.51 | 282.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 277.05 | 280.65 | 281.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 281.70 | 280.67 | 281.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 11:15:00 | 281.70 | 280.67 | 281.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 281.70 | 280.67 | 281.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 281.70 | 280.67 | 281.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 280.50 | 280.63 | 281.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 278.00 | 280.65 | 281.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 286.50 | 277.72 | 276.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 286.50 | 277.72 | 276.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 286.90 | 280.93 | 278.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 277.00 | 281.16 | 279.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 277.00 | 281.16 | 279.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 277.00 | 281.16 | 279.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 277.00 | 281.16 | 279.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 278.70 | 280.67 | 279.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 279.05 | 280.44 | 279.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 273.65 | 279.14 | 279.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 273.65 | 279.14 | 279.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 270.85 | 275.60 | 277.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 281.45 | 276.03 | 277.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 281.45 | 276.03 | 277.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 281.45 | 276.03 | 277.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 281.65 | 276.03 | 277.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 278.50 | 276.53 | 277.33 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 282.40 | 278.42 | 278.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 282.45 | 279.77 | 278.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 275.90 | 279.49 | 278.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 275.90 | 279.49 | 278.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 275.90 | 279.49 | 278.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 275.90 | 279.49 | 278.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 276.70 | 278.93 | 278.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 278.10 | 279.06 | 278.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 15:15:00 | 305.91 | 298.74 | 293.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 330.60 | 333.87 | 334.29 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 336.75 | 334.54 | 334.34 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 331.40 | 334.47 | 334.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 329.55 | 332.01 | 333.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 322.20 | 318.37 | 322.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 322.20 | 318.37 | 322.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 322.20 | 318.37 | 322.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 322.35 | 318.37 | 322.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 323.20 | 319.34 | 322.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 322.90 | 319.34 | 322.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 321.80 | 319.83 | 322.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 319.80 | 319.75 | 322.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 331.60 | 321.51 | 321.99 | SL hit (close>static) qty=1.00 sl=323.35 alert=retest2 |

### Cycle 108 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 330.50 | 323.31 | 322.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 12:15:00 | 334.45 | 325.54 | 323.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 09:15:00 | 328.35 | 329.23 | 326.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:45:00 | 328.90 | 329.23 | 326.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 327.45 | 328.87 | 327.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 327.45 | 328.87 | 327.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 326.50 | 328.39 | 327.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 326.50 | 328.39 | 327.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 328.00 | 328.32 | 327.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 336.75 | 328.32 | 327.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 333.10 | 329.27 | 327.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 342.95 | 336.11 | 333.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 342.80 | 339.53 | 336.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:30:00 | 342.45 | 340.57 | 337.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:30:00 | 342.85 | 341.47 | 338.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 338.20 | 341.11 | 338.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 338.20 | 341.11 | 338.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 336.15 | 340.12 | 338.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 336.15 | 340.12 | 338.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 335.40 | 339.18 | 338.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 335.40 | 339.18 | 338.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 335.70 | 337.74 | 337.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 335.70 | 337.74 | 337.83 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 339.75 | 337.95 | 337.88 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 336.55 | 338.05 | 338.06 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 343.55 | 338.98 | 338.47 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-05-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 09:15:00 | 331.30 | 338.66 | 338.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 10:15:00 | 325.80 | 336.09 | 337.72 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-11-21 09:15:00 | 487.25 | 2024-11-25 14:15:00 | 530.09 | TARGET_HIT | 1.00 | 8.79% |
| BUY | retest2 | 2024-11-21 09:45:00 | 481.90 | 2024-11-25 14:15:00 | 529.43 | TARGET_HIT | 1.00 | 9.86% |
| BUY | retest2 | 2024-11-22 12:00:00 | 481.30 | 2024-11-26 09:15:00 | 535.98 | TARGET_HIT | 1.00 | 11.36% |
| BUY | retest2 | 2024-12-10 09:15:00 | 530.70 | 2024-12-12 09:15:00 | 528.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-10 10:45:00 | 526.70 | 2024-12-12 09:15:00 | 528.05 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-12-12 09:15:00 | 530.00 | 2024-12-12 09:15:00 | 528.05 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-12-20 15:00:00 | 521.50 | 2024-12-23 09:15:00 | 532.35 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-01-03 10:15:00 | 542.25 | 2025-01-06 12:15:00 | 533.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-01-06 11:15:00 | 540.05 | 2025-01-06 12:15:00 | 533.25 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-01-09 11:45:00 | 518.00 | 2025-01-13 09:15:00 | 492.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:45:00 | 518.00 | 2025-01-13 14:15:00 | 466.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-20 12:45:00 | 523.80 | 2025-01-21 10:15:00 | 510.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-01-20 13:30:00 | 524.00 | 2025-01-21 10:15:00 | 510.60 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-01-20 14:00:00 | 523.90 | 2025-01-21 10:15:00 | 510.60 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-01-20 15:15:00 | 525.00 | 2025-01-21 10:15:00 | 510.60 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-01-23 12:45:00 | 499.65 | 2025-01-27 09:15:00 | 474.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:45:00 | 499.65 | 2025-01-27 10:15:00 | 449.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 428.95 | 2025-02-18 15:15:00 | 434.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-02-17 09:45:00 | 429.20 | 2025-02-18 15:15:00 | 434.70 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-02-17 10:45:00 | 429.00 | 2025-02-18 15:15:00 | 434.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-02-17 13:45:00 | 428.40 | 2025-02-18 15:15:00 | 434.70 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-02-25 09:30:00 | 457.00 | 2025-02-27 10:15:00 | 446.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-03-26 09:15:00 | 479.15 | 2025-03-26 12:15:00 | 475.30 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-03-26 10:00:00 | 479.25 | 2025-03-26 12:15:00 | 475.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-03-26 10:45:00 | 480.05 | 2025-03-26 12:15:00 | 475.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-04-04 10:00:00 | 462.25 | 2025-04-07 09:15:00 | 416.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 13:00:00 | 446.95 | 2025-04-22 12:15:00 | 443.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-04-21 10:30:00 | 446.30 | 2025-04-22 15:15:00 | 442.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-21 11:15:00 | 446.00 | 2025-04-22 15:15:00 | 442.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-04-21 12:30:00 | 446.20 | 2025-04-22 15:15:00 | 442.40 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-22 09:15:00 | 448.30 | 2025-04-22 15:15:00 | 442.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-04-30 09:45:00 | 421.80 | 2025-05-07 11:15:00 | 426.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-30 12:45:00 | 421.65 | 2025-05-07 11:15:00 | 426.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-05-02 10:45:00 | 420.60 | 2025-05-07 11:15:00 | 426.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-02 13:30:00 | 420.95 | 2025-05-07 11:15:00 | 426.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-05-06 14:45:00 | 412.80 | 2025-05-07 11:15:00 | 426.60 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-05-16 10:45:00 | 464.15 | 2025-05-21 11:15:00 | 455.45 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2025-05-23 09:30:00 | 446.70 | 2025-05-29 09:15:00 | 424.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-23 09:30:00 | 446.70 | 2025-05-30 11:15:00 | 425.45 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-05-27 13:15:00 | 434.50 | 2025-06-06 09:15:00 | 428.55 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2025-06-17 09:15:00 | 441.50 | 2025-06-17 10:15:00 | 445.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-25 09:15:00 | 440.05 | 2025-06-26 10:15:00 | 435.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-25 14:30:00 | 439.10 | 2025-06-26 10:15:00 | 435.65 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-26 09:15:00 | 440.90 | 2025-06-26 10:15:00 | 435.65 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-04 11:15:00 | 428.75 | 2025-07-10 13:15:00 | 426.75 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-07-07 09:15:00 | 428.05 | 2025-07-10 13:15:00 | 426.75 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-07-10 09:45:00 | 428.70 | 2025-07-10 13:15:00 | 426.75 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-07-18 14:15:00 | 420.40 | 2025-07-22 09:15:00 | 425.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-31 10:00:00 | 396.10 | 2025-08-01 11:15:00 | 406.20 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-08-11 09:15:00 | 420.15 | 2025-08-26 10:15:00 | 428.35 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-09-03 11:45:00 | 439.65 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2025-09-04 09:45:00 | 440.10 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-09-04 15:15:00 | 440.50 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2025-09-05 09:45:00 | 443.10 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-09-08 09:30:00 | 440.65 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-09-08 11:45:00 | 439.50 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2025-09-08 12:45:00 | 439.60 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-09-09 09:30:00 | 439.75 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2025-09-10 09:15:00 | 442.10 | 2025-09-19 10:15:00 | 450.40 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-09-29 11:30:00 | 447.20 | 2025-09-30 14:15:00 | 455.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-29 12:30:00 | 447.15 | 2025-09-30 14:15:00 | 455.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-09-29 15:00:00 | 445.00 | 2025-09-30 14:15:00 | 455.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-09-30 12:15:00 | 445.20 | 2025-09-30 14:15:00 | 455.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-10-24 10:30:00 | 443.20 | 2025-10-28 13:15:00 | 447.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-24 13:30:00 | 443.30 | 2025-10-28 13:15:00 | 447.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-27 12:00:00 | 443.30 | 2025-10-28 13:15:00 | 447.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-27 12:30:00 | 443.25 | 2025-10-28 13:15:00 | 447.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-31 09:30:00 | 453.75 | 2025-10-31 14:15:00 | 448.60 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-31 11:45:00 | 454.10 | 2025-10-31 14:15:00 | 448.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-11-03 10:00:00 | 453.85 | 2025-11-03 11:15:00 | 449.55 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-10 12:00:00 | 438.25 | 2025-11-11 13:15:00 | 416.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 12:00:00 | 438.25 | 2025-11-12 11:15:00 | 421.40 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-12-08 10:30:00 | 395.70 | 2025-12-09 14:15:00 | 401.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-15 13:15:00 | 399.10 | 2025-12-16 09:15:00 | 395.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-15 15:00:00 | 399.30 | 2025-12-16 09:15:00 | 395.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-22 12:45:00 | 381.00 | 2025-12-24 10:15:00 | 385.85 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-13 14:15:00 | 360.75 | 2026-01-20 12:15:00 | 342.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 360.75 | 2026-01-20 14:15:00 | 350.15 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2026-01-13 15:15:00 | 358.10 | 2026-01-21 09:15:00 | 340.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 358.10 | 2026-01-22 09:15:00 | 347.10 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2026-02-11 11:15:00 | 338.40 | 2026-02-11 13:15:00 | 338.40 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 314.50 | 2026-02-25 09:15:00 | 298.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 314.50 | 2026-02-25 14:15:00 | 308.75 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2026-03-16 09:15:00 | 274.00 | 2026-03-17 10:15:00 | 280.50 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-03-20 15:15:00 | 278.00 | 2026-03-25 09:15:00 | 286.50 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2026-03-27 11:30:00 | 279.05 | 2026-03-30 09:15:00 | 273.65 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-04-02 11:30:00 | 278.10 | 2026-04-08 15:15:00 | 305.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 12:30:00 | 319.80 | 2026-04-28 10:15:00 | 331.60 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-05-04 11:15:00 | 342.95 | 2026-05-06 13:15:00 | 335.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-05-05 09:15:00 | 342.80 | 2026-05-06 13:15:00 | 335.70 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-05-05 13:30:00 | 342.45 | 2026-05-06 13:15:00 | 335.70 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-05-05 14:30:00 | 342.85 | 2026-05-06 13:15:00 | 335.70 | STOP_HIT | 1.00 | -2.09% |
