# International Gemmological Institute (India) Ltd. (IGIL)

## Backtest Summary

- **Window:** 2024-12-20 09:15:00 → 2026-05-11 15:15:00 (2389 bars)
- **Last close:** 345.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 113 |
| ALERT1 | 75 |
| ALERT2 | 73 |
| ALERT2_SKIP | 41 |
| ALERT3 | 196 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 15 |
| TARGET_HIT | 4 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 38
- **Target hits / Stop hits / Partials:** 4 / 62 / 15
- **Avg / median % per leg:** 1.28% / 0.19%
- **Sum % (uncompounded):** 104.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 8 | 24.2% | 2 | 31 | 0 | -0.39% | -12.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.50% | -2.5% |
| BUY @ 3rd Alert (retest2) | 32 | 8 | 25.0% | 2 | 30 | 0 | -0.32% | -10.2% |
| SELL (all) | 48 | 35 | 72.9% | 2 | 31 | 15 | 2.43% | 116.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 35 | 72.9% | 2 | 31 | 15 | 2.43% | 116.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.50% | -2.5% |
| retest2 (combined) | 80 | 43 | 53.8% | 4 | 61 | 15 | 1.33% | 106.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 587.50 | 596.07 | 596.69 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 604.80 | 597.81 | 597.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 618.00 | 611.23 | 606.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 12:15:00 | 613.30 | 616.89 | 612.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 12:15:00 | 613.30 | 616.89 | 612.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 613.30 | 616.89 | 612.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 609.45 | 616.89 | 612.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 605.60 | 614.63 | 611.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 605.60 | 614.63 | 611.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 604.00 | 612.51 | 611.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 604.00 | 612.51 | 611.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 602.00 | 610.41 | 610.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 589.60 | 610.41 | 610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 583.65 | 605.05 | 607.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 560.75 | 584.10 | 594.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 553.05 | 545.44 | 553.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 553.05 | 545.44 | 553.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 553.05 | 545.44 | 553.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 553.05 | 545.44 | 553.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 549.10 | 546.17 | 553.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 544.20 | 546.02 | 552.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 548.00 | 546.38 | 551.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 545.55 | 546.38 | 551.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 15:00:00 | 546.65 | 546.44 | 551.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 13:15:00 | 516.99 | 532.03 | 541.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 13:15:00 | 520.60 | 532.03 | 541.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 13:15:00 | 518.27 | 532.03 | 541.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 13:15:00 | 519.32 | 532.03 | 541.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 525.20 | 524.39 | 534.17 | SL hit (close>ema200) qty=0.50 sl=524.39 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 549.70 | 523.34 | 522.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 553.55 | 533.02 | 527.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 11:15:00 | 536.25 | 541.76 | 536.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 11:15:00 | 536.25 | 541.76 | 536.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 536.25 | 541.76 | 536.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 536.25 | 541.76 | 536.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 530.15 | 539.44 | 535.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 530.15 | 539.44 | 535.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 525.00 | 536.55 | 534.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 525.00 | 536.55 | 534.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 521.00 | 531.43 | 532.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 483.90 | 521.93 | 528.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 490.65 | 490.03 | 503.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 490.65 | 490.03 | 503.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 505.45 | 493.11 | 503.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 501.70 | 493.11 | 503.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 503.30 | 495.15 | 503.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:15:00 | 509.75 | 495.15 | 503.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 500.00 | 496.12 | 503.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 514.15 | 496.12 | 503.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 510.00 | 499.02 | 503.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 510.00 | 499.02 | 503.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 517.65 | 502.75 | 504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 514.00 | 502.75 | 504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 509.20 | 506.00 | 505.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 520.00 | 512.32 | 509.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 506.75 | 512.35 | 510.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 506.75 | 512.35 | 510.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 506.75 | 512.35 | 510.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 506.75 | 512.35 | 510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 508.40 | 511.56 | 510.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:45:00 | 508.00 | 511.56 | 510.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 506.25 | 510.04 | 509.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 512.00 | 510.04 | 509.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 509.85 | 510.00 | 509.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:30:00 | 512.10 | 510.47 | 509.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 13:15:00 | 507.00 | 509.69 | 509.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 507.00 | 509.69 | 509.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 09:15:00 | 502.65 | 507.71 | 508.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 516.65 | 503.59 | 505.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 13:15:00 | 516.65 | 503.59 | 505.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 516.65 | 503.59 | 505.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:45:00 | 512.30 | 503.59 | 505.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 514.50 | 505.77 | 506.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 509.05 | 507.02 | 507.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 508.35 | 507.19 | 507.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 13:15:00 | 508.50 | 507.25 | 507.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 13:15:00 | 508.50 | 507.25 | 507.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 15:15:00 | 510.20 | 508.12 | 507.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 11:15:00 | 523.40 | 524.43 | 518.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 12:00:00 | 523.40 | 524.43 | 518.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 516.10 | 522.58 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:45:00 | 520.65 | 522.58 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 507.95 | 519.65 | 517.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 507.95 | 519.65 | 517.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 500.00 | 515.72 | 515.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 497.00 | 505.79 | 510.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 504.40 | 503.96 | 507.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:30:00 | 503.55 | 503.96 | 507.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 475.70 | 462.85 | 467.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 475.70 | 462.85 | 467.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 471.15 | 464.51 | 467.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 463.35 | 468.25 | 468.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 440.18 | 452.76 | 459.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-19 09:15:00 | 417.02 | 431.19 | 437.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 445.85 | 438.04 | 437.08 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 420.25 | 433.79 | 435.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 14:15:00 | 419.85 | 426.94 | 431.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 420.00 | 418.42 | 424.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 13:00:00 | 420.00 | 418.42 | 424.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 419.25 | 417.41 | 421.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 421.05 | 417.41 | 421.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 404.20 | 401.92 | 407.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:30:00 | 406.40 | 401.92 | 407.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 406.85 | 402.19 | 405.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 409.30 | 402.19 | 405.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 409.00 | 403.55 | 405.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 366.25 | 403.55 | 405.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 347.94 | 365.66 | 381.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-05 09:15:00 | 329.62 | 336.62 | 356.19 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 310.20 | 306.10 | 305.75 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 09:15:00 | 298.05 | 305.14 | 305.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-18 09:15:00 | 294.00 | 301.66 | 303.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 12:15:00 | 306.10 | 301.03 | 302.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 12:15:00 | 306.10 | 301.03 | 302.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 306.10 | 301.03 | 302.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 306.10 | 301.03 | 302.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 315.70 | 303.96 | 303.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 335.05 | 314.57 | 309.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 375.95 | 381.07 | 371.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 375.95 | 381.07 | 371.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 375.20 | 380.03 | 372.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 374.45 | 380.03 | 372.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 371.10 | 378.24 | 372.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 370.65 | 378.24 | 372.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 380.60 | 378.71 | 373.41 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 358.55 | 369.55 | 370.57 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 378.45 | 371.71 | 371.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 390.50 | 375.31 | 373.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 382.85 | 383.92 | 379.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 382.85 | 383.92 | 379.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 382.85 | 383.92 | 379.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 390.75 | 384.38 | 380.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 390.25 | 384.38 | 380.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:45:00 | 392.30 | 385.57 | 381.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 377.50 | 384.55 | 382.28 | SL hit (close<static) qty=1.00 sl=377.75 alert=retest2 |

### Cycle 17 — SELL (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 13:15:00 | 377.60 | 380.42 | 380.78 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 393.25 | 382.99 | 381.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 396.00 | 387.55 | 384.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 389.00 | 393.53 | 389.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 389.00 | 393.53 | 389.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 389.00 | 393.53 | 389.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 391.15 | 393.53 | 389.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 389.90 | 392.80 | 389.38 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 382.35 | 387.16 | 387.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 363.25 | 381.61 | 384.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 333.50 | 333.33 | 344.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 14:15:00 | 338.30 | 334.48 | 340.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 338.30 | 334.48 | 340.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:45:00 | 338.55 | 334.48 | 340.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 342.95 | 335.90 | 340.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:30:00 | 354.15 | 335.90 | 340.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 343.30 | 337.38 | 340.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 11:45:00 | 341.70 | 338.21 | 340.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 14:00:00 | 340.60 | 339.32 | 340.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 10:15:00 | 353.30 | 341.38 | 340.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 353.30 | 341.38 | 340.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 363.95 | 352.22 | 346.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 367.50 | 370.01 | 364.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 367.50 | 370.01 | 364.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 367.50 | 369.08 | 366.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 362.80 | 369.08 | 366.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 367.70 | 368.68 | 366.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 366.90 | 368.68 | 366.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 366.75 | 368.30 | 366.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 366.30 | 368.30 | 366.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 362.80 | 367.20 | 366.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 362.80 | 367.20 | 366.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 363.70 | 366.50 | 365.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:15:00 | 362.80 | 366.50 | 365.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 363.60 | 365.32 | 365.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 354.10 | 362.88 | 364.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 357.25 | 354.73 | 358.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 357.25 | 354.73 | 358.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 355.40 | 354.86 | 358.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 355.40 | 354.86 | 358.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 358.00 | 355.49 | 358.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:00:00 | 352.60 | 354.86 | 357.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 349.60 | 352.50 | 355.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 334.97 | 339.88 | 342.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 13:15:00 | 332.12 | 338.26 | 341.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 332.65 | 330.94 | 335.28 | SL hit (close>ema200) qty=0.50 sl=330.94 alert=retest2 |

### Cycle 22 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 355.05 | 340.01 | 338.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 358.95 | 346.19 | 341.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 346.35 | 348.65 | 344.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 346.35 | 348.65 | 344.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 343.95 | 347.71 | 344.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 337.55 | 347.71 | 344.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 341.30 | 346.43 | 344.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 345.60 | 344.38 | 343.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 09:15:00 | 380.16 | 372.95 | 364.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 376.90 | 381.07 | 381.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 373.00 | 377.01 | 378.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 375.30 | 374.51 | 376.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 375.30 | 374.51 | 376.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 375.30 | 374.51 | 376.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 376.65 | 374.51 | 376.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 376.35 | 374.88 | 376.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 376.35 | 374.88 | 376.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 376.85 | 375.27 | 376.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 378.55 | 375.27 | 376.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 376.85 | 375.59 | 376.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 376.80 | 375.59 | 376.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 376.50 | 375.77 | 376.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 377.65 | 375.77 | 376.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 375.25 | 375.67 | 376.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 376.65 | 375.67 | 376.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 377.00 | 375.83 | 376.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 373.85 | 375.21 | 376.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:30:00 | 373.85 | 374.86 | 375.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 15:15:00 | 372.60 | 374.99 | 375.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 373.00 | 370.44 | 370.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 373.00 | 370.44 | 370.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 377.50 | 371.85 | 370.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 14:15:00 | 384.00 | 385.14 | 381.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 384.00 | 385.14 | 381.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 383.15 | 384.74 | 381.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 380.60 | 384.74 | 381.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 388.90 | 385.58 | 381.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 391.25 | 386.34 | 382.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 390.95 | 387.26 | 383.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 390.00 | 388.35 | 384.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 391.65 | 388.55 | 385.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 407.45 | 410.54 | 404.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 413.35 | 404.87 | 403.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 402.30 | 405.51 | 405.15 | SL hit (close<static) qty=1.00 sl=403.85 alert=retest2 |

### Cycle 25 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 402.90 | 404.59 | 404.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 13:15:00 | 401.95 | 404.06 | 404.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 406.95 | 403.40 | 403.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 406.95 | 403.40 | 403.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 406.95 | 403.40 | 403.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 407.00 | 403.40 | 403.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 405.35 | 403.79 | 404.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 406.85 | 403.79 | 404.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 405.05 | 404.04 | 404.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 406.00 | 404.04 | 404.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 381.05 | 385.61 | 390.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 376.55 | 381.43 | 385.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 13:15:00 | 357.72 | 364.75 | 372.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 347.90 | 347.10 | 352.75 | SL hit (close>ema200) qty=0.50 sl=347.10 alert=retest2 |

### Cycle 26 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 356.30 | 352.07 | 351.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 361.25 | 354.37 | 352.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 356.80 | 357.01 | 355.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:45:00 | 357.05 | 357.01 | 355.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 369.60 | 373.10 | 369.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 369.00 | 373.10 | 369.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 368.95 | 372.27 | 369.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 368.95 | 372.27 | 369.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 367.90 | 371.40 | 369.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:30:00 | 367.90 | 371.40 | 369.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 368.65 | 370.85 | 369.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 368.20 | 370.85 | 369.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 366.85 | 368.58 | 368.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 363.40 | 368.58 | 368.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 366.50 | 368.16 | 368.29 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 376.10 | 369.06 | 368.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 380.20 | 372.45 | 370.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 15:15:00 | 390.00 | 390.29 | 386.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 392.35 | 390.29 | 386.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 391.15 | 390.73 | 388.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 391.00 | 390.73 | 388.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 395.40 | 391.54 | 389.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 388.70 | 391.54 | 389.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 390.50 | 390.73 | 389.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 390.65 | 390.73 | 389.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 391.35 | 390.74 | 389.81 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 384.90 | 388.93 | 389.11 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 390.35 | 388.20 | 388.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 395.50 | 390.03 | 389.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 396.65 | 396.75 | 393.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 396.65 | 396.75 | 393.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 396.65 | 396.75 | 393.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 393.75 | 396.75 | 393.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 395.70 | 397.22 | 395.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 395.70 | 397.22 | 395.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 396.40 | 397.05 | 395.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 399.25 | 397.05 | 395.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 393.70 | 396.13 | 395.46 | SL hit (close<static) qty=1.00 sl=395.25 alert=retest2 |

### Cycle 31 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 389.95 | 394.04 | 394.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 388.30 | 392.90 | 393.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 391.45 | 390.98 | 392.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 13:00:00 | 391.45 | 390.98 | 392.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 389.55 | 390.54 | 391.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 391.00 | 390.54 | 391.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 390.80 | 390.05 | 391.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 390.50 | 390.05 | 391.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 397.00 | 391.44 | 391.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 395.90 | 391.44 | 391.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 396.55 | 392.46 | 392.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 404.45 | 396.34 | 394.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 404.85 | 409.73 | 405.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 404.85 | 409.73 | 405.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 404.85 | 409.73 | 405.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 404.85 | 409.73 | 405.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 405.35 | 408.85 | 405.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 402.75 | 408.85 | 405.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 405.00 | 408.08 | 405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 405.00 | 408.08 | 405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 405.00 | 407.47 | 405.47 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 399.00 | 403.72 | 404.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 395.90 | 402.16 | 403.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 399.30 | 398.65 | 400.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 399.30 | 398.65 | 400.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 401.80 | 399.28 | 400.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 401.80 | 399.28 | 400.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 404.60 | 400.34 | 401.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 407.10 | 400.34 | 401.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 408.00 | 401.87 | 401.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 421.50 | 405.80 | 403.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 414.55 | 414.85 | 410.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 402.65 | 414.85 | 410.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 400.30 | 411.94 | 409.17 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 397.70 | 406.93 | 407.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 393.70 | 402.05 | 404.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 381.95 | 381.43 | 390.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 381.95 | 381.43 | 390.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 375.40 | 377.08 | 383.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 378.75 | 377.08 | 383.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 335.25 | 334.88 | 342.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 339.50 | 334.88 | 342.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 339.00 | 335.53 | 339.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 339.00 | 335.53 | 339.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 334.80 | 335.39 | 338.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 334.65 | 335.41 | 338.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 15:15:00 | 340.50 | 338.43 | 338.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 340.50 | 338.43 | 338.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 342.60 | 339.48 | 338.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 339.20 | 339.42 | 338.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 339.20 | 339.42 | 338.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 339.80 | 339.58 | 339.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 333.55 | 339.58 | 339.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 334.80 | 338.62 | 338.76 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 344.00 | 339.02 | 338.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 347.05 | 340.63 | 339.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 350.75 | 353.47 | 349.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 350.75 | 353.47 | 349.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 350.75 | 353.47 | 349.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 350.75 | 353.47 | 349.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 350.35 | 352.85 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 349.75 | 352.85 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 347.85 | 351.85 | 349.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 347.85 | 351.85 | 349.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 353.45 | 352.17 | 350.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 355.50 | 352.77 | 350.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 347.50 | 350.64 | 350.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 347.50 | 350.64 | 350.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 345.25 | 349.56 | 350.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 342.85 | 337.59 | 340.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 342.85 | 337.59 | 340.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 342.85 | 337.59 | 340.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 342.85 | 337.59 | 340.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 344.95 | 339.06 | 340.95 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 345.00 | 342.13 | 342.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 347.90 | 344.54 | 343.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 343.30 | 345.29 | 344.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 343.30 | 345.29 | 344.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 343.30 | 345.29 | 344.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 343.30 | 345.29 | 344.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 344.65 | 345.16 | 344.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 343.55 | 345.16 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 343.55 | 344.84 | 344.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 345.65 | 344.84 | 344.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 345.75 | 347.20 | 346.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 342.65 | 345.73 | 345.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 342.65 | 345.73 | 345.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 342.15 | 345.01 | 345.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 344.65 | 343.92 | 344.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 344.65 | 343.92 | 344.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 344.65 | 343.92 | 344.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 344.65 | 343.92 | 344.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 349.20 | 344.98 | 345.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 349.20 | 344.98 | 345.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 348.50 | 345.68 | 345.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 352.00 | 346.95 | 345.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 346.35 | 349.46 | 348.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 346.35 | 349.46 | 348.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 346.35 | 349.46 | 348.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 346.35 | 349.46 | 348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 345.10 | 348.59 | 347.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 346.55 | 348.16 | 347.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 346.40 | 347.59 | 347.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 347.05 | 347.48 | 347.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 347.05 | 347.48 | 347.49 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 347.75 | 347.53 | 347.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 350.00 | 348.03 | 347.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 365.55 | 368.31 | 362.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:45:00 | 366.70 | 368.31 | 362.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 365.65 | 367.21 | 363.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 365.65 | 367.21 | 363.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 372.35 | 367.73 | 364.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 384.50 | 373.67 | 370.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 391.05 | 374.90 | 372.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 378.80 | 382.43 | 380.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:15:00 | 378.20 | 380.11 | 379.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 375.75 | 379.24 | 379.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 375.75 | 379.24 | 379.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 371.60 | 377.71 | 378.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 15:15:00 | 372.90 | 371.67 | 374.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 15:15:00 | 372.90 | 371.67 | 374.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 372.90 | 371.67 | 374.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 362.00 | 367.24 | 370.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:00:00 | 360.95 | 365.54 | 368.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 361.75 | 363.50 | 366.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 361.20 | 363.60 | 366.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 343.90 | 347.11 | 351.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 343.66 | 347.11 | 351.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 342.90 | 345.96 | 350.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 343.14 | 345.96 | 350.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 346.50 | 345.77 | 349.66 | SL hit (close>ema200) qty=0.50 sl=345.77 alert=retest2 |

### Cycle 46 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 357.05 | 348.62 | 348.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 359.25 | 353.11 | 350.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 355.75 | 355.76 | 353.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:00:00 | 355.75 | 355.76 | 353.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 352.70 | 355.63 | 353.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 352.10 | 355.63 | 353.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 351.85 | 354.87 | 353.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 352.00 | 354.87 | 353.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 351.15 | 352.98 | 353.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 350.70 | 351.64 | 352.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 348.30 | 346.57 | 348.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 348.30 | 346.57 | 348.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 348.30 | 346.57 | 348.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 347.60 | 346.57 | 348.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 347.80 | 346.81 | 348.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 348.00 | 346.81 | 348.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 347.20 | 346.89 | 347.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:45:00 | 347.65 | 346.89 | 347.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 348.45 | 347.20 | 348.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 348.45 | 347.20 | 348.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 348.00 | 347.36 | 348.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:15:00 | 349.25 | 347.36 | 348.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 346.50 | 347.19 | 347.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:15:00 | 348.35 | 347.19 | 347.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 348.35 | 347.42 | 347.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 350.70 | 347.42 | 347.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 350.75 | 348.09 | 348.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:15:00 | 354.60 | 348.09 | 348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 348.15 | 348.10 | 348.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 347.95 | 347.92 | 348.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 345.50 | 340.74 | 340.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 342.15 | 341.02 | 340.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 342.15 | 341.02 | 340.95 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 339.00 | 340.93 | 341.02 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 345.00 | 341.74 | 341.38 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 336.25 | 340.39 | 340.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 11:15:00 | 334.80 | 339.27 | 340.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 334.40 | 333.82 | 335.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 341.00 | 335.25 | 336.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 341.00 | 335.25 | 336.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 341.00 | 335.25 | 336.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 337.00 | 335.60 | 336.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 352.85 | 335.60 | 336.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 350.75 | 338.63 | 337.67 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 339.00 | 341.20 | 341.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 334.00 | 339.76 | 340.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 339.70 | 338.38 | 339.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 339.70 | 338.38 | 339.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 339.70 | 338.38 | 339.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 340.90 | 338.38 | 339.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 340.20 | 338.74 | 339.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 340.20 | 338.74 | 339.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 340.65 | 339.13 | 339.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 339.80 | 339.13 | 339.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 337.50 | 337.63 | 338.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 337.40 | 337.46 | 338.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 338.80 | 337.73 | 338.37 | SL hit (close>static) qty=1.00 sl=338.60 alert=retest2 |

### Cycle 54 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 339.60 | 338.70 | 338.65 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 337.75 | 338.61 | 338.62 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 338.70 | 338.63 | 338.63 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 338.50 | 338.60 | 338.62 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 339.20 | 338.72 | 338.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 340.50 | 339.08 | 338.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 337.35 | 338.73 | 338.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 337.35 | 338.73 | 338.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 337.35 | 338.73 | 338.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 337.35 | 338.73 | 338.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 335.85 | 338.16 | 338.44 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 339.45 | 338.60 | 338.58 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 337.00 | 338.28 | 338.44 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 343.70 | 339.35 | 338.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 348.85 | 341.25 | 339.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 339.65 | 342.01 | 340.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 339.65 | 342.01 | 340.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 339.65 | 342.01 | 340.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 339.65 | 342.01 | 340.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 338.35 | 341.27 | 340.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 339.40 | 341.27 | 340.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 336.90 | 339.80 | 340.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 335.20 | 338.88 | 339.65 | Break + close below crossover candle low |

### Cycle 64 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 357.90 | 341.52 | 340.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 363.30 | 345.88 | 342.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 355.10 | 358.51 | 351.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 355.00 | 358.36 | 355.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 355.00 | 358.36 | 355.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 355.50 | 358.36 | 355.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 349.15 | 356.52 | 354.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 349.15 | 356.52 | 354.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 350.60 | 355.33 | 354.13 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 13:15:00 | 349.45 | 353.19 | 353.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 14:15:00 | 345.80 | 351.71 | 352.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 347.30 | 346.74 | 348.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 347.30 | 346.74 | 348.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 347.30 | 346.74 | 348.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 348.35 | 346.74 | 348.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 349.00 | 347.19 | 348.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 348.75 | 347.19 | 348.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 347.20 | 347.19 | 348.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 346.60 | 347.12 | 348.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 346.70 | 347.18 | 348.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:15:00 | 329.27 | 336.23 | 340.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:15:00 | 329.36 | 336.23 | 340.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 334.35 | 325.14 | 329.71 | SL hit (close>ema200) qty=0.50 sl=325.14 alert=retest2 |

### Cycle 66 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 336.00 | 329.32 | 329.10 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 325.60 | 329.35 | 329.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 321.15 | 327.20 | 328.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 322.15 | 322.08 | 324.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 324.05 | 322.33 | 323.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 324.05 | 322.33 | 323.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 324.05 | 322.33 | 323.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 322.00 | 322.27 | 323.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:30:00 | 322.00 | 322.27 | 323.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 319.95 | 321.80 | 323.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 326.85 | 321.80 | 323.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 326.40 | 322.72 | 323.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 326.65 | 322.72 | 323.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 326.10 | 323.40 | 323.87 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 328.05 | 324.33 | 324.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 329.55 | 327.90 | 327.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 15:15:00 | 328.00 | 328.01 | 327.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 332.25 | 328.01 | 327.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 329.20 | 329.93 | 328.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 329.20 | 329.93 | 328.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 329.00 | 329.74 | 328.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 327.90 | 329.74 | 328.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 323.95 | 328.58 | 328.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 323.95 | 328.58 | 328.35 | SL hit (close<ema400) qty=1.00 sl=328.35 alert=retest1 |

### Cycle 69 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 324.45 | 327.76 | 328.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 323.10 | 326.82 | 327.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 326.50 | 325.99 | 327.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 326.50 | 325.99 | 327.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 325.20 | 325.83 | 326.84 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 328.35 | 327.21 | 327.11 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 324.70 | 326.85 | 327.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 323.10 | 325.56 | 326.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 13:15:00 | 313.45 | 313.00 | 317.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 14:00:00 | 313.45 | 313.00 | 317.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 315.00 | 313.40 | 317.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 316.20 | 313.40 | 317.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 314.00 | 313.90 | 316.65 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 325.65 | 317.96 | 317.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 327.95 | 321.16 | 319.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 10:15:00 | 321.55 | 322.43 | 320.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:45:00 | 321.50 | 322.43 | 320.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 321.40 | 322.16 | 321.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 321.65 | 322.16 | 321.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 322.80 | 322.28 | 321.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:30:00 | 323.20 | 322.09 | 321.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 320.95 | 321.86 | 321.56 | SL hit (close<static) qty=1.00 sl=321.20 alert=retest2 |

### Cycle 73 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 317.80 | 320.73 | 321.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 10:15:00 | 317.30 | 320.04 | 320.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 322.30 | 320.49 | 320.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 322.30 | 320.49 | 320.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 322.30 | 320.49 | 320.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 322.30 | 320.49 | 320.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 321.80 | 320.75 | 320.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 321.40 | 320.75 | 320.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 321.85 | 321.17 | 321.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 321.85 | 321.17 | 321.13 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 319.05 | 320.75 | 320.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 317.90 | 319.44 | 320.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 319.50 | 319.32 | 319.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:45:00 | 318.75 | 319.32 | 319.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 319.00 | 319.35 | 319.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 318.00 | 319.35 | 319.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 322.15 | 317.19 | 316.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 322.15 | 317.19 | 316.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 328.90 | 320.68 | 318.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 327.25 | 328.27 | 325.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 327.30 | 328.27 | 325.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 331.05 | 334.60 | 332.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 331.45 | 334.60 | 332.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 331.00 | 333.88 | 332.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 331.15 | 333.88 | 332.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 330.60 | 332.07 | 332.09 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 334.70 | 332.22 | 332.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 335.45 | 332.86 | 332.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 332.25 | 333.20 | 332.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 332.25 | 333.20 | 332.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 332.25 | 333.20 | 332.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 332.25 | 333.20 | 332.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 332.10 | 332.98 | 332.61 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 330.20 | 332.11 | 332.26 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 334.20 | 332.41 | 332.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 335.50 | 333.54 | 332.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 329.60 | 332.75 | 332.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 329.60 | 332.75 | 332.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 329.60 | 332.75 | 332.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 330.70 | 332.75 | 332.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 334.00 | 333.00 | 332.69 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 327.60 | 332.03 | 332.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 326.65 | 330.95 | 331.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 14:15:00 | 317.15 | 315.30 | 319.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:30:00 | 318.55 | 315.30 | 319.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 310.20 | 309.97 | 311.98 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 314.00 | 312.45 | 312.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 314.50 | 313.08 | 312.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 314.00 | 314.10 | 313.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 309.30 | 314.10 | 313.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 310.20 | 313.32 | 313.09 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 310.25 | 312.70 | 312.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 306.40 | 310.61 | 311.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 304.55 | 298.91 | 302.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 304.55 | 298.91 | 302.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 304.55 | 298.91 | 302.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 304.55 | 298.91 | 302.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 303.80 | 299.89 | 302.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:00:00 | 301.30 | 300.17 | 302.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 318.00 | 297.81 | 297.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 318.00 | 297.81 | 297.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 323.50 | 316.81 | 311.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 317.75 | 317.99 | 313.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:15:00 | 316.40 | 317.99 | 313.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 312.40 | 316.88 | 313.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 312.40 | 316.88 | 313.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 310.75 | 315.65 | 313.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 310.75 | 315.65 | 313.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 310.90 | 313.05 | 312.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 307.85 | 313.05 | 312.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 312.35 | 312.91 | 312.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 313.05 | 313.43 | 312.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 310.05 | 312.13 | 312.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 310.05 | 312.13 | 312.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 304.35 | 310.39 | 311.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 308.55 | 308.35 | 309.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 308.55 | 308.35 | 309.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 308.55 | 308.35 | 309.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 332.40 | 308.35 | 309.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 330.85 | 312.85 | 311.67 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 315.05 | 323.07 | 323.82 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 333.75 | 323.04 | 321.84 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 324.65 | 328.14 | 328.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 322.10 | 325.43 | 326.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 326.20 | 324.15 | 325.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 326.20 | 324.15 | 325.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 326.20 | 324.15 | 325.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 326.20 | 324.15 | 325.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 328.15 | 324.95 | 325.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 328.15 | 324.95 | 325.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 331.05 | 326.60 | 326.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 332.00 | 329.61 | 328.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 14:15:00 | 331.30 | 331.66 | 329.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 15:00:00 | 331.30 | 331.66 | 329.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 333.10 | 331.91 | 330.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 334.65 | 331.91 | 330.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 333.95 | 333.18 | 331.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 334.30 | 332.49 | 331.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 335.00 | 332.80 | 331.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 327.85 | 331.81 | 331.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 327.85 | 331.81 | 331.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 327.60 | 330.97 | 331.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 327.60 | 330.97 | 331.08 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 15:15:00 | 332.30 | 331.12 | 331.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 338.35 | 332.56 | 331.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 334.50 | 334.96 | 333.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 334.50 | 334.96 | 333.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 333.75 | 334.61 | 333.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 326.70 | 334.61 | 333.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 324.65 | 332.62 | 332.68 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 332.15 | 331.36 | 331.26 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 328.60 | 330.81 | 331.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 324.75 | 329.07 | 330.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 11:15:00 | 329.00 | 328.98 | 329.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:45:00 | 329.00 | 328.98 | 329.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 329.40 | 329.07 | 329.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 329.40 | 329.07 | 329.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 330.00 | 329.25 | 329.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 330.05 | 329.25 | 329.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 330.00 | 329.40 | 329.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 330.00 | 329.40 | 329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 331.00 | 329.72 | 329.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 319.10 | 329.72 | 329.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 331.20 | 325.05 | 324.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 331.20 | 325.05 | 324.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 332.75 | 326.59 | 325.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 328.20 | 330.69 | 328.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 328.20 | 330.69 | 328.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 328.20 | 330.69 | 328.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 326.35 | 330.69 | 328.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 328.45 | 330.24 | 328.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 326.35 | 330.24 | 328.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 330.10 | 330.21 | 328.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:45:00 | 331.90 | 330.79 | 329.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 332.75 | 331.82 | 330.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 332.35 | 331.23 | 330.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 332.95 | 331.00 | 330.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 331.30 | 331.30 | 330.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 331.30 | 331.30 | 330.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 330.00 | 331.04 | 330.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:30:00 | 330.15 | 331.04 | 330.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 330.05 | 330.84 | 330.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 330.25 | 330.84 | 330.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 328.40 | 330.35 | 330.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 328.40 | 330.35 | 330.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 327.00 | 329.68 | 330.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 305.30 | 303.93 | 310.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 305.30 | 303.93 | 310.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 304.35 | 304.01 | 309.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 315.80 | 307.48 | 310.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 317.60 | 309.50 | 311.51 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 320.10 | 313.99 | 313.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 324.10 | 316.01 | 314.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 315.10 | 316.30 | 314.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 315.10 | 316.30 | 314.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 315.10 | 316.30 | 314.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 11:00:00 | 321.60 | 317.36 | 315.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 12:15:00 | 319.00 | 327.18 | 327.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 319.00 | 327.18 | 327.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 14:15:00 | 318.10 | 325.06 | 326.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 326.80 | 324.11 | 325.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 326.80 | 324.11 | 325.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 326.80 | 324.11 | 325.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 326.80 | 324.11 | 325.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 326.00 | 324.49 | 325.78 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 332.50 | 326.60 | 326.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 335.60 | 328.74 | 327.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 330.95 | 331.50 | 329.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 330.95 | 331.50 | 329.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 330.95 | 331.50 | 329.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 324.55 | 331.50 | 329.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 323.15 | 329.83 | 329.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 322.85 | 329.83 | 329.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 321.25 | 328.11 | 328.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 320.30 | 325.46 | 327.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 319.75 | 318.30 | 321.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 319.75 | 318.30 | 321.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 319.75 | 318.30 | 321.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 319.75 | 318.30 | 321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 327.65 | 319.90 | 321.67 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 327.00 | 323.04 | 322.81 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 315.60 | 322.25 | 322.61 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 326.95 | 320.36 | 320.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 333.25 | 324.82 | 322.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 335.50 | 338.32 | 335.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 335.50 | 338.32 | 335.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 335.50 | 338.32 | 335.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 336.60 | 338.32 | 335.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 13:15:00 | 370.26 | 357.75 | 350.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 357.50 | 361.12 | 361.57 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 363.20 | 361.89 | 361.79 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 355.40 | 360.53 | 361.20 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 362.05 | 361.32 | 361.25 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 358.95 | 360.85 | 361.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 358.40 | 360.36 | 360.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 349.90 | 349.73 | 353.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 349.90 | 349.73 | 353.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 349.90 | 349.73 | 353.28 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 358.65 | 354.57 | 354.24 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 343.35 | 352.44 | 353.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 341.30 | 344.81 | 347.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 13:15:00 | 345.25 | 344.90 | 347.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 14:00:00 | 345.25 | 344.90 | 347.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 348.40 | 345.60 | 347.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 348.40 | 345.60 | 347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 349.70 | 346.42 | 347.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 351.30 | 346.42 | 347.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 357.25 | 348.59 | 348.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 356.85 | 348.59 | 348.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 356.50 | 350.17 | 349.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 362.35 | 356.89 | 353.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 358.55 | 362.03 | 359.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 358.55 | 362.03 | 359.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 358.55 | 362.03 | 359.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 359.05 | 362.03 | 359.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 354.20 | 360.47 | 358.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 354.20 | 360.47 | 358.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 357.35 | 359.84 | 358.70 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 355.20 | 357.84 | 357.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 352.90 | 356.33 | 357.19 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-16 11:45:00 | 544.20 | 2025-01-17 13:15:00 | 516.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:45:00 | 548.00 | 2025-01-17 13:15:00 | 520.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 14:15:00 | 545.55 | 2025-01-17 13:15:00 | 518.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 15:00:00 | 546.65 | 2025-01-17 13:15:00 | 519.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:45:00 | 544.20 | 2025-01-20 10:15:00 | 525.20 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-01-16 13:45:00 | 548.00 | 2025-01-20 10:15:00 | 525.20 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-01-16 14:15:00 | 545.55 | 2025-01-20 10:15:00 | 525.20 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-01-16 15:00:00 | 546.65 | 2025-01-20 10:15:00 | 525.20 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-01-20 13:00:00 | 519.25 | 2025-01-23 09:15:00 | 549.70 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2025-01-21 11:15:00 | 515.70 | 2025-01-23 09:15:00 | 549.70 | STOP_HIT | 1.00 | -6.59% |
| SELL | retest2 | 2025-01-22 10:15:00 | 519.35 | 2025-01-23 09:15:00 | 549.70 | STOP_HIT | 1.00 | -5.84% |
| BUY | retest2 | 2025-01-31 11:30:00 | 512.10 | 2025-01-31 13:15:00 | 507.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-02-03 09:15:00 | 509.05 | 2025-02-03 13:15:00 | 508.50 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-02-03 10:45:00 | 508.35 | 2025-02-03 13:15:00 | 508.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-02-14 09:15:00 | 463.35 | 2025-02-17 09:15:00 | 440.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 463.35 | 2025-02-19 09:15:00 | 417.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 09:15:00 | 366.25 | 2025-03-04 09:15:00 | 347.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-03 09:15:00 | 366.25 | 2025-03-05 09:15:00 | 329.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-01 12:45:00 | 390.75 | 2025-04-02 09:15:00 | 377.50 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-04-01 13:15:00 | 390.25 | 2025-04-02 09:15:00 | 377.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-04-01 13:45:00 | 392.30 | 2025-04-02 09:15:00 | 377.50 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-04-15 11:45:00 | 341.70 | 2025-04-17 10:15:00 | 353.30 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-04-15 14:00:00 | 340.60 | 2025-04-17 10:15:00 | 353.30 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-04-28 14:00:00 | 352.60 | 2025-05-06 12:15:00 | 334.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 09:30:00 | 349.60 | 2025-05-06 13:15:00 | 332.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 14:00:00 | 352.60 | 2025-05-07 13:15:00 | 332.65 | STOP_HIT | 0.50 | 5.66% |
| SELL | retest2 | 2025-04-29 09:30:00 | 349.60 | 2025-05-07 13:15:00 | 332.65 | STOP_HIT | 0.50 | 4.85% |
| BUY | retest2 | 2025-05-09 15:15:00 | 345.60 | 2025-05-14 09:15:00 | 380.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-26 11:45:00 | 373.85 | 2025-05-30 09:15:00 | 373.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-05-26 13:30:00 | 373.85 | 2025-05-30 09:15:00 | 373.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-05-26 15:15:00 | 372.60 | 2025-05-30 09:15:00 | 373.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-06-03 11:15:00 | 391.25 | 2025-06-10 10:15:00 | 402.30 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-06-03 12:00:00 | 390.95 | 2025-06-10 12:15:00 | 402.90 | STOP_HIT | 1.00 | 3.06% |
| BUY | retest2 | 2025-06-03 14:00:00 | 390.00 | 2025-06-10 12:15:00 | 402.90 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2025-06-03 14:45:00 | 391.65 | 2025-06-10 12:15:00 | 402.90 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-06-09 10:15:00 | 413.35 | 2025-06-10 12:15:00 | 402.90 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-06-17 09:45:00 | 376.55 | 2025-06-18 13:15:00 | 357.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:45:00 | 376.55 | 2025-06-23 09:15:00 | 347.90 | STOP_HIT | 0.50 | 7.61% |
| BUY | retest2 | 2025-07-17 11:15:00 | 399.25 | 2025-07-17 12:15:00 | 393.70 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-13 11:45:00 | 334.65 | 2025-08-14 15:15:00 | 340.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-08-25 12:15:00 | 355.50 | 2025-08-26 13:15:00 | 347.50 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-03 09:15:00 | 345.65 | 2025-09-04 14:15:00 | 342.65 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-04 09:30:00 | 345.75 | 2025-09-04 14:15:00 | 342.65 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-09 11:45:00 | 346.55 | 2025-09-09 13:15:00 | 347.05 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-09-09 13:15:00 | 346.40 | 2025-09-09 13:15:00 | 347.05 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-16 09:15:00 | 384.50 | 2025-09-19 12:15:00 | 375.75 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-09-17 09:15:00 | 391.05 | 2025-09-19 12:15:00 | 375.75 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-09-18 15:00:00 | 378.80 | 2025-09-19 12:15:00 | 375.75 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-19 12:15:00 | 378.20 | 2025-09-19 12:15:00 | 375.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-24 09:15:00 | 362.00 | 2025-09-29 12:15:00 | 343.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 11:00:00 | 360.95 | 2025-09-29 12:15:00 | 343.66 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-09-24 13:45:00 | 361.75 | 2025-09-29 14:15:00 | 342.90 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-09-24 15:15:00 | 361.20 | 2025-09-29 14:15:00 | 343.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 362.00 | 2025-09-30 09:15:00 | 346.50 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2025-09-24 11:00:00 | 360.95 | 2025-09-30 09:15:00 | 346.50 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 361.75 | 2025-09-30 09:15:00 | 346.50 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-09-24 15:15:00 | 361.20 | 2025-09-30 09:15:00 | 346.50 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-10-13 11:45:00 | 347.95 | 2025-10-16 09:15:00 | 342.15 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2025-10-16 09:15:00 | 345.50 | 2025-10-16 09:15:00 | 342.15 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-10-29 09:30:00 | 337.40 | 2025-10-29 10:15:00 | 338.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-12 12:45:00 | 346.60 | 2025-11-18 10:15:00 | 329.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 346.70 | 2025-11-18 10:15:00 | 329.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 12:45:00 | 346.60 | 2025-11-19 13:15:00 | 334.35 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-11-12 14:15:00 | 346.70 | 2025-11-19 13:15:00 | 334.35 | STOP_HIT | 0.50 | 3.56% |
| BUY | retest1 | 2025-12-01 09:15:00 | 332.25 | 2025-12-02 09:15:00 | 323.95 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-12-12 13:30:00 | 323.20 | 2025-12-12 14:15:00 | 320.95 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-15 13:15:00 | 321.40 | 2025-12-15 14:15:00 | 321.85 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-12-17 14:15:00 | 318.00 | 2025-12-22 13:15:00 | 322.15 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-22 12:00:00 | 301.30 | 2026-01-28 09:15:00 | 318.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2026-02-01 10:45:00 | 313.05 | 2026-02-01 14:15:00 | 310.05 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-19 10:15:00 | 334.65 | 2026-02-20 12:15:00 | 327.60 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-02-19 13:15:00 | 333.95 | 2026-02-20 12:15:00 | 327.60 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-20 09:45:00 | 334.30 | 2026-02-20 12:15:00 | 327.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-20 10:30:00 | 335.00 | 2026-02-20 12:15:00 | 327.60 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-03-02 09:15:00 | 319.10 | 2026-03-05 14:15:00 | 331.20 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-03-09 14:45:00 | 331.90 | 2026-03-11 14:15:00 | 328.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-03-10 10:00:00 | 332.75 | 2026-03-11 14:15:00 | 328.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-10 12:30:00 | 332.35 | 2026-03-11 14:15:00 | 328.40 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-03-11 09:15:00 | 332.95 | 2026-03-11 14:15:00 | 328.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-03-18 11:00:00 | 321.60 | 2026-03-23 12:15:00 | 319.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-04-13 10:15:00 | 336.60 | 2026-04-16 13:15:00 | 370.26 | TARGET_HIT | 1.00 | 10.00% |
