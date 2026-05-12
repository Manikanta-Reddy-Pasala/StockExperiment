# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 50 |
| ALERT2 | 47 |
| ALERT2_SKIP | 28 |
| ALERT3 | 113 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 15 / 31
- **Target hits / Stop hits / Partials:** 0 / 42 / 4
- **Avg / median % per leg:** -0.09% / -0.89%
- **Sum % (uncompounded):** -3.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 0 | 16 | 0 | -0.64% | -10.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 3 | 18.8% | 0 | 16 | 0 | -0.64% | -10.2% |
| SELL (all) | 30 | 12 | 40.0% | 0 | 26 | 4 | 0.21% | 6.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 12 | 40.0% | 0 | 26 | 4 | 0.21% | 6.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 15 | 32.6% | 0 | 42 | 4 | -0.09% | -4.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 461.35 | 443.13 | 441.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 464.00 | 454.33 | 448.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 498.10 | 519.07 | 507.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 502.30 | 515.72 | 506.81 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 494.60 | 503.13 | 503.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 491.00 | 496.00 | 499.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 509.35 | 488.64 | 492.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 499.00 | 490.72 | 493.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 501.00 | 490.72 | 493.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 499.20 | 494.43 | 494.32 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 491.05 | 493.98 | 494.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 487.50 | 490.43 | 492.11 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 507.55 | 493.86 | 493.51 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 492.00 | 495.29 | 495.64 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 502.00 | 496.37 | 496.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 517.15 | 500.53 | 497.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 520.00 | 520.88 | 515.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 520.00 | 520.88 | 515.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 514.65 | 520.34 | 517.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 514.65 | 520.34 | 517.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 517.95 | 519.86 | 517.94 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 511.70 | 516.05 | 516.62 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 533.95 | 518.43 | 517.43 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 517.45 | 524.76 | 525.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 14:15:00 | 514.00 | 522.61 | 524.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 521.45 | 521.21 | 523.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 503.95 | 502.16 | 506.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 509.70 | 502.16 | 506.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 504.15 | 502.71 | 505.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 500.20 | 502.33 | 505.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 500.20 | 501.92 | 504.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:45:00 | 500.10 | 501.71 | 504.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 498.00 | 502.16 | 504.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 498.00 | 501.32 | 503.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 502.15 | 501.34 | 503.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 500.45 | 501.16 | 503.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 503.20 | 501.16 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 495.80 | 499.26 | 501.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 498.10 | 499.26 | 501.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 501.30 | 499.67 | 501.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 501.30 | 499.67 | 501.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 497.00 | 499.13 | 501.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 503.50 | 499.13 | 501.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 499.60 | 499.23 | 501.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 503.90 | 499.23 | 501.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 493.00 | 488.83 | 490.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 494.05 | 488.83 | 490.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 493.10 | 489.69 | 491.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 493.10 | 489.69 | 491.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 491.85 | 490.12 | 491.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 499.30 | 490.12 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 512.65 | 500.55 | 497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 510.30 | 511.02 | 506.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 510.80 | 511.02 | 506.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 502.45 | 511.00 | 509.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 502.45 | 511.00 | 509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 503.95 | 509.59 | 508.94 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 502.75 | 508.23 | 508.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 495.85 | 502.61 | 504.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 495.10 | 492.71 | 495.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 494.50 | 493.34 | 495.71 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 499.00 | 496.41 | 496.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 506.65 | 500.19 | 498.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 502.90 | 502.92 | 500.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 495.15 | 502.92 | 500.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 499.55 | 502.24 | 500.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 497.15 | 502.24 | 500.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 518.50 | 505.49 | 502.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 527.25 | 513.03 | 506.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 527.40 | 513.03 | 506.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 531.35 | 535.32 | 537.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 519.95 | 517.65 | 524.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 518.45 | 517.65 | 524.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 515.55 | 516.53 | 520.63 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 528.30 | 522.49 | 521.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 591.85 | 542.18 | 532.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 570.50 | 575.35 | 555.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 570.50 | 575.35 | 555.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 557.60 | 570.92 | 556.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 557.60 | 570.92 | 556.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 556.05 | 567.94 | 556.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 555.70 | 567.94 | 556.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 552.35 | 564.83 | 556.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 552.35 | 564.83 | 556.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 554.35 | 562.73 | 556.12 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 531.70 | 549.40 | 551.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 525.15 | 544.55 | 548.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 520.00 | 518.26 | 523.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 520.00 | 518.26 | 523.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 521.85 | 518.98 | 523.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 516.15 | 518.98 | 523.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 505.00 | 507.80 | 508.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 502.40 | 506.72 | 507.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 501.10 | 504.48 | 505.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 476.05 | 481.83 | 486.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 472.30 | 467.82 | 472.84 | SL hit (close>ema200) qty=0.50 sl=467.82 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 478.45 | 474.33 | 474.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 481.95 | 476.70 | 475.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 478.75 | 478.81 | 476.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 478.75 | 478.81 | 476.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 479.00 | 478.85 | 477.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 484.40 | 478.85 | 477.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 521.30 | 519.63 | 519.54 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 518.20 | 519.35 | 519.42 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 520.20 | 519.52 | 519.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 521.60 | 520.09 | 519.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 519.15 | 519.90 | 519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 519.90 | 519.90 | 519.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 517.50 | 519.90 | 519.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 522.30 | 520.38 | 519.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 527.70 | 521.85 | 520.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 527.30 | 522.88 | 521.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 526.65 | 523.30 | 521.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 526.45 | 523.30 | 521.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 521.65 | 523.23 | 521.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 521.65 | 523.23 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 521.40 | 522.87 | 521.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 521.40 | 522.87 | 521.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 521.80 | 522.65 | 521.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 515.60 | 522.65 | 521.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 506.70 | 515.20 | 518.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 517.55 | 512.63 | 516.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 510.85 | 512.27 | 515.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 508.50 | 511.38 | 514.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 510.00 | 510.65 | 513.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 509.85 | 507.87 | 509.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 514.00 | 511.27 | 510.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 518.00 | 518.68 | 516.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 521.15 | 518.68 | 516.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 514.95 | 517.95 | 516.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 514.10 | 517.95 | 516.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 517.80 | 517.92 | 516.62 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 510.80 | 515.43 | 515.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 510.00 | 514.35 | 515.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 515.55 | 514.48 | 515.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 518.20 | 515.22 | 515.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 522.55 | 515.22 | 515.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 521.20 | 516.42 | 515.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 12:15:00 | 534.40 | 520.01 | 517.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:30:00 | 517.85 | 519.42 | 517.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 517.00 | 518.93 | 517.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 519.45 | 518.93 | 517.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 508.60 | 516.95 | 519.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 517.30 | 514.08 | 516.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 519.45 | 515.15 | 516.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 519.70 | 515.15 | 516.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 520.85 | 516.29 | 517.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 529.60 | 516.29 | 517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 517.90 | 517.49 | 517.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 517.90 | 517.49 | 517.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 518.80 | 517.75 | 517.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 519.50 | 517.75 | 517.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 520.20 | 518.24 | 518.04 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 516.50 | 517.89 | 517.90 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 518.90 | 518.03 | 517.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 520.60 | 518.66 | 518.26 | Break + close above crossover candle high |

### Cycle 32 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 512.20 | 517.46 | 517.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 509.60 | 515.89 | 517.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 520.00 | 516.32 | 516.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 520.10 | 517.08 | 517.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 521.00 | 517.08 | 517.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 521.85 | 518.03 | 517.64 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 514.55 | 517.66 | 517.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 512.50 | 516.63 | 517.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 512.35 | 510.43 | 512.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 513.40 | 511.03 | 512.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 513.40 | 511.03 | 512.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 512.50 | 511.32 | 512.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:15:00 | 511.65 | 511.32 | 512.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 523.95 | 516.22 | 514.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 516.50 | 517.65 | 516.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 515.90 | 517.30 | 516.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 516.45 | 517.30 | 516.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 515.50 | 516.94 | 516.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 518.60 | 516.94 | 516.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 520.65 | 518.36 | 517.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 517.70 | 518.36 | 517.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 585.85 | 586.25 | 582.04 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 567.85 | 579.50 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 564.65 | 576.53 | 579.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 566.75 | 539.69 | 547.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 565.50 | 544.85 | 549.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 565.50 | 544.85 | 549.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 539.50 | 546.47 | 549.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:45:00 | 535.10 | 543.90 | 547.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:15:00 | 508.34 | 516.02 | 520.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 517.70 | 511.94 | 515.18 | SL hit (close>ema200) qty=0.50 sl=511.94 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 519.80 | 516.27 | 516.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 525.80 | 520.00 | 518.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 519.20 | 520.08 | 518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 517.50 | 519.57 | 518.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 517.50 | 519.57 | 518.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 514.70 | 518.59 | 518.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 514.70 | 518.59 | 518.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 514.30 | 517.06 | 517.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 509.20 | 515.49 | 516.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 528.85 | 503.33 | 505.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 519.00 | 506.46 | 506.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 523.00 | 506.46 | 506.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 516.05 | 508.38 | 507.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 522.50 | 512.80 | 509.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 536.40 | 537.59 | 528.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 536.40 | 537.59 | 528.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 529.90 | 535.09 | 531.34 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 526.80 | 529.71 | 529.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 524.60 | 528.69 | 529.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 521.90 | 521.58 | 524.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 523.50 | 521.58 | 524.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 524.40 | 522.14 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 525.25 | 522.14 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 524.00 | 522.51 | 524.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 524.70 | 522.51 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 524.25 | 522.86 | 524.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:30:00 | 524.95 | 522.86 | 524.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 524.00 | 523.09 | 524.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 523.30 | 523.24 | 524.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 523.70 | 524.80 | 524.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 522.80 | 524.40 | 524.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 519.80 | 520.82 | 522.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 520.00 | 520.28 | 521.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 535.10 | 524.68 | 523.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 542.40 | 546.53 | 540.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:00:00 | 542.40 | 546.53 | 540.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 536.35 | 544.49 | 540.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 536.35 | 544.49 | 540.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 533.25 | 542.24 | 539.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 533.20 | 542.24 | 539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 533.80 | 538.32 | 538.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 533.20 | 538.32 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 533.55 | 537.36 | 537.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 532.15 | 535.78 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 530.80 | 534.45 | 535.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 531.05 | 533.07 | 534.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 539.25 | 533.44 | 531.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 549.00 | 549.33 | 543.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 551.00 | 549.33 | 543.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 602.75 | 589.64 | 574.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 617.65 | 596.41 | 585.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 612.55 | 624.54 | 626.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 579.20 | 559.86 | 568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 580.00 | 563.89 | 569.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 572.55 | 565.62 | 570.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:45:00 | 573.70 | 569.81 | 570.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 582.80 | 576.00 | 573.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 576.20 | 581.19 | 578.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 575.05 | 579.97 | 577.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 572.80 | 579.97 | 577.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 570.80 | 578.13 | 577.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 570.80 | 578.13 | 577.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 566.50 | 575.81 | 576.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 562.25 | 573.09 | 575.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 538.40 | 536.82 | 541.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 549.00 | 539.26 | 542.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 548.50 | 539.26 | 542.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 546.30 | 540.67 | 542.82 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 561.80 | 544.89 | 544.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 570.55 | 555.54 | 550.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 536.45 | 553.29 | 554.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 533.15 | 549.27 | 552.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 547.60 | 545.99 | 549.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 547.60 | 545.99 | 549.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 541.85 | 545.50 | 548.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 531.15 | 545.50 | 548.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:30:00 | 533.40 | 525.45 | 531.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 538.55 | 531.91 | 532.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 548.65 | 538.86 | 536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 541.70 | 541.73 | 538.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 541.70 | 541.73 | 538.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 534.75 | 540.22 | 538.76 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 535.15 | 537.76 | 538.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 525.70 | 535.35 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 533.65 | 533.56 | 535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 553.45 | 537.45 | 536.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 561.80 | 551.12 | 545.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 539.15 | 552.99 | 549.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 539.60 | 550.31 | 548.31 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 533.00 | 544.32 | 545.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 526.50 | 537.22 | 541.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 524.80 | 532.20 | 535.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 525.10 | 529.77 | 531.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 558.20 | 540.61 | 536.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 548.85 | 549.00 | 543.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 548.85 | 549.00 | 543.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 543.80 | 547.99 | 544.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.60 | 547.99 | 544.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 553.90 | 549.17 | 545.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 569.35 | 553.93 | 551.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 553.80 | 568.90 | 572.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 544.50 | 543.99 | 551.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 544.50 | 543.99 | 551.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 532.10 | 521.25 | 524.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 535.50 | 521.25 | 524.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 533.10 | 523.62 | 525.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 533.10 | 523.62 | 525.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 525.40 | 526.07 | 526.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 516.05 | 526.07 | 526.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 527.50 | 524.95 | 525.39 | SL hit (close>static) qty=1.00 sl=526.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 511.10 | 502.19 | 501.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 512.45 | 505.55 | 502.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 497.00 | 505.58 | 503.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 497.35 | 503.93 | 503.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 501.00 | 503.35 | 503.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 491.00 | 499.74 | 501.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 503.00 | 497.54 | 499.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 501.75 | 498.39 | 499.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 502.00 | 498.39 | 499.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 499.95 | 499.80 | 500.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 497.50 | 499.49 | 500.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 472.62 | 490.36 | 495.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 486.10 | 482.87 | 488.70 | SL hit (close>ema200) qty=0.50 sl=482.87 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 499.80 | 490.15 | 489.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 506.25 | 497.21 | 493.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 542.85 | 547.76 | 530.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:00:00 | 542.85 | 547.76 | 530.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 538.25 | 544.57 | 535.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 571.10 | 544.57 | 535.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 545.55 | 553.08 | 549.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 545.00 | 552.09 | 548.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 543.30 | 547.07 | 548.55 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 560.10 | 549.68 | 549.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 566.75 | 553.09 | 551.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 561.95 | 565.57 | 560.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 561.95 | 565.57 | 560.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 559.45 | 564.34 | 560.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 559.60 | 564.34 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 561.00 | 563.68 | 560.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 559.15 | 563.68 | 560.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 559.00 | 562.74 | 560.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 564.30 | 563.12 | 560.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 564.95 | 560.58 | 560.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 566.30 | 561.72 | 560.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 625.10 | 641.00 | 627.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 631.55 | 639.11 | 627.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 648.55 | 636.79 | 627.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 676.00 | 659.80 | 658.48 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 656.05 | 661.14 | 661.39 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 664.10 | 661.47 | 661.26 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 658.75 | 660.98 | 661.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 588.55 | 646.50 | 654.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 605.70 | 603.01 | 621.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 605.70 | 603.01 | 621.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 617.45 | 606.95 | 618.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 617.45 | 606.95 | 618.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 614.75 | 608.51 | 618.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 609.75 | 610.69 | 617.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-17 12:00:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 12:30:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 13:45:00 | 500.10 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-17 15:15:00 | 498.00 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-07-14 13:30:00 | 527.25 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-07-14 14:15:00 | 527.40 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-08-08 09:15:00 | 516.15 | 2025-08-18 13:15:00 | 514.90 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-08-26 15:15:00 | 476.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-09-01 09:15:00 | 472.30 | STOP_HIT | 0.50 | 5.75% |
| BUY | retest2 | 2025-09-03 09:15:00 | 484.40 | 2025-09-22 15:15:00 | 517.45 | STOP_HIT | 1.00 | 6.82% |
| BUY | retest2 | 2025-09-25 09:45:00 | 527.70 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-25 10:30:00 | 527.30 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-25 11:45:00 | 526.65 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-09-25 12:15:00 | 526.45 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-29 11:30:00 | 508.50 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-29 14:45:00 | 510.00 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-01 10:45:00 | 509.85 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-09 09:15:00 | 519.45 | 2025-10-10 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-23 10:15:00 | 511.65 | 2025-10-23 13:15:00 | 517.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-17 09:15:00 | 508.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-18 11:15:00 | 517.70 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-12-03 14:15:00 | 523.30 | 2025-12-04 09:15:00 | 531.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-08 10:30:00 | 519.80 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-08 15:15:00 | 520.00 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-12-16 12:00:00 | 530.80 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-17 09:30:00 | 531.05 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-31 09:15:00 | 617.65 | 2026-01-07 15:15:00 | 616.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-14 12:00:00 | 572.55 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-16 10:45:00 | 573.70 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-01 12:15:00 | 531.15 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-03 09:30:00 | 533.40 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-02-03 12:30:00 | 538.55 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-13 15:15:00 | 524.80 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-02-17 14:30:00 | 525.10 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-02-24 09:15:00 | 569.35 | 2026-03-02 11:15:00 | 563.45 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-03-12 09:15:00 | 516.05 | 2026-03-12 12:15:00 | 527.50 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-13 15:15:00 | 495.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-16 14:15:00 | 497.50 | STOP_HIT | 0.50 | 4.56% |
| BUY | retest2 | 2026-03-19 12:00:00 | 501.00 | 2026-03-19 12:15:00 | 496.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-23 10:15:00 | 472.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-24 09:15:00 | 486.10 | STOP_HIT | 0.50 | 2.29% |
| BUY | retest2 | 2026-04-01 09:15:00 | 571.10 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2026-04-02 11:45:00 | 545.55 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-02 12:30:00 | 545.00 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-10 09:30:00 | 564.30 | 2026-04-13 09:15:00 | 553.05 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-04-21 09:15:00 | 648.55 | 2026-04-24 13:15:00 | 648.20 | STOP_HIT | 1.00 | -0.05% |
