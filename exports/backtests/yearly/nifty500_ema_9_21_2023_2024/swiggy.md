# Swiggy Ltd. (SWIGGY)

## Backtest Summary

- **Window:** 2024-11-13 09:15:00 → 2026-05-08 15:15:00 (2557 bars)
- **Last close:** 282.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 99 |
| ALERT1 | 67 |
| ALERT2 | 67 |
| ALERT2_SKIP | 34 |
| ALERT3 | 164 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 91 |
| PARTIAL | 26 |
| TARGET_HIT | 9 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 64 / 49
- **Target hits / Stop hits / Partials:** 9 / 78 / 26
- **Avg / median % per leg:** 1.66% / 1.77%
- **Sum % (uncompounded):** 187.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 10 | 30.3% | 5 | 28 | 0 | 0.16% | 5.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 10 | 30.3% | 5 | 28 | 0 | 0.16% | 5.2% |
| SELL (all) | 80 | 54 | 67.5% | 4 | 50 | 26 | 2.28% | 182.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 80 | 54 | 67.5% | 4 | 50 | 26 | 2.28% | 182.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 113 | 64 | 56.6% | 9 | 78 | 26 | 1.66% | 187.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 433.80 | 422.16 | 420.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 444.50 | 430.91 | 425.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 484.75 | 485.70 | 468.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 486.85 | 485.70 | 468.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 471.20 | 484.60 | 475.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 471.20 | 484.60 | 475.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 469.80 | 481.64 | 475.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:45:00 | 467.65 | 481.64 | 475.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 470.45 | 471.90 | 471.93 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 476.80 | 472.88 | 472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 480.45 | 474.70 | 473.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 14:15:00 | 498.95 | 501.95 | 491.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 498.95 | 501.95 | 491.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 490.10 | 499.58 | 491.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 516.95 | 499.58 | 491.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 11:15:00 | 568.65 | 532.81 | 515.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 519.70 | 535.95 | 537.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 512.30 | 523.65 | 529.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 512.35 | 511.70 | 518.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 512.35 | 511.70 | 518.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 512.35 | 511.70 | 518.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 512.35 | 511.70 | 518.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 526.00 | 515.03 | 518.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 526.00 | 515.03 | 518.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 530.20 | 518.06 | 519.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:15:00 | 533.00 | 518.06 | 519.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 533.00 | 521.05 | 520.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 547.30 | 526.30 | 523.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 584.40 | 585.23 | 567.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 584.40 | 585.23 | 567.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 578.15 | 579.74 | 573.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:15:00 | 576.60 | 579.74 | 573.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 576.60 | 579.11 | 573.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:15:00 | 572.75 | 579.11 | 573.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 590.80 | 581.45 | 574.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 599.45 | 583.12 | 578.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:30:00 | 599.40 | 595.26 | 588.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 573.10 | 586.67 | 587.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 573.10 | 586.67 | 587.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 10:15:00 | 570.15 | 583.36 | 585.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 562.50 | 553.54 | 560.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 562.50 | 553.54 | 560.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 562.50 | 553.54 | 560.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 562.50 | 553.54 | 560.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 558.35 | 554.50 | 560.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:00:00 | 553.70 | 554.93 | 559.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 556.55 | 549.03 | 548.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 556.55 | 549.03 | 548.33 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 543.65 | 548.51 | 548.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 534.00 | 542.97 | 545.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 15:15:00 | 540.00 | 534.13 | 538.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 15:15:00 | 540.00 | 534.13 | 538.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 540.00 | 534.13 | 538.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 523.70 | 534.13 | 538.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 09:45:00 | 525.00 | 533.42 | 538.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 524.10 | 533.42 | 538.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 497.52 | 513.34 | 524.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 498.75 | 513.34 | 524.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 497.89 | 513.34 | 524.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 501.80 | 498.06 | 509.30 | SL hit (close>ema200) qty=0.50 sl=498.06 alert=retest2 |

### Cycle 9 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 489.00 | 482.55 | 481.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 499.05 | 486.72 | 483.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 483.30 | 486.70 | 484.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 11:15:00 | 483.30 | 486.70 | 484.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 483.30 | 486.70 | 484.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 483.30 | 486.70 | 484.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 481.00 | 485.56 | 484.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 480.45 | 485.56 | 484.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 485.00 | 484.44 | 483.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 477.45 | 484.44 | 483.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 485.50 | 484.65 | 483.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 477.20 | 484.65 | 483.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 472.50 | 482.22 | 482.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 442.65 | 470.42 | 475.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 433.20 | 430.39 | 444.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 14:45:00 | 431.85 | 430.39 | 444.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 450.20 | 434.95 | 443.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 447.85 | 434.95 | 443.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 452.00 | 438.36 | 444.59 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 459.50 | 449.45 | 448.51 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 446.40 | 448.67 | 448.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 421.60 | 443.26 | 446.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 410.45 | 409.29 | 421.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:00:00 | 410.45 | 409.29 | 421.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 421.90 | 412.20 | 419.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 421.90 | 412.20 | 419.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 420.75 | 413.91 | 419.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 415.55 | 414.66 | 419.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 415.45 | 414.67 | 418.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 429.00 | 417.54 | 419.53 | SL hit (close>static) qty=1.00 sl=422.60 alert=retest2 |

### Cycle 13 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 425.25 | 420.82 | 420.79 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 417.65 | 420.19 | 420.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 11:15:00 | 410.45 | 418.24 | 419.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 416.00 | 411.74 | 415.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 416.00 | 411.74 | 415.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 416.00 | 411.74 | 415.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 416.00 | 411.74 | 415.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 416.15 | 412.62 | 415.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:15:00 | 418.40 | 412.62 | 415.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 415.80 | 413.26 | 415.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 12:30:00 | 412.30 | 412.96 | 415.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:30:00 | 413.30 | 415.27 | 415.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 430.75 | 418.37 | 416.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 430.75 | 418.37 | 416.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 443.80 | 423.45 | 419.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 438.35 | 446.18 | 436.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:30:00 | 442.40 | 446.18 | 436.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 435.00 | 442.87 | 436.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:45:00 | 434.05 | 442.87 | 436.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 431.85 | 440.67 | 436.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 429.80 | 440.67 | 436.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 432.60 | 437.21 | 435.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 424.40 | 437.21 | 435.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 417.90 | 433.34 | 434.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 403.80 | 419.56 | 425.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 344.50 | 341.86 | 355.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 347.85 | 341.86 | 355.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 354.50 | 344.61 | 351.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 354.65 | 344.61 | 351.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 362.50 | 348.18 | 352.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 362.50 | 348.18 | 352.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 366.10 | 356.60 | 355.53 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 337.35 | 353.36 | 354.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 332.60 | 349.21 | 352.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 342.75 | 340.98 | 346.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 15:00:00 | 342.75 | 340.98 | 346.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 342.80 | 341.44 | 345.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 332.15 | 339.20 | 342.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 347.35 | 343.26 | 342.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 347.35 | 343.26 | 342.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 364.85 | 347.58 | 344.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 371.50 | 373.30 | 366.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 09:15:00 | 369.15 | 373.30 | 366.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 354.85 | 369.61 | 365.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 354.85 | 369.61 | 365.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 365.65 | 368.82 | 365.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 370.00 | 368.21 | 365.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 358.85 | 364.06 | 364.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 358.85 | 364.06 | 364.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 13:15:00 | 347.95 | 352.09 | 356.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 344.65 | 331.55 | 336.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 344.65 | 331.55 | 336.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 344.65 | 331.55 | 336.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 344.65 | 331.55 | 336.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 341.50 | 333.54 | 336.77 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 348.35 | 339.79 | 339.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 354.90 | 346.52 | 342.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 358.50 | 360.15 | 355.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 358.50 | 360.15 | 355.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 358.50 | 360.15 | 355.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 361.65 | 360.88 | 356.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:00:00 | 362.00 | 361.23 | 358.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:30:00 | 362.35 | 361.29 | 358.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:30:00 | 362.20 | 361.07 | 358.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 359.85 | 360.83 | 358.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:15:00 | 358.10 | 360.83 | 358.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 358.30 | 360.32 | 358.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 351.95 | 358.54 | 358.45 | SL hit (close<static) qty=1.00 sl=353.55 alert=retest2 |

### Cycle 22 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 353.40 | 357.51 | 357.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 347.60 | 353.01 | 355.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 350.65 | 350.60 | 352.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 350.65 | 350.60 | 352.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 350.65 | 350.60 | 352.74 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 356.75 | 353.68 | 353.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 365.35 | 358.00 | 356.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 360.45 | 360.87 | 358.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 14:00:00 | 360.45 | 360.87 | 358.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 350.30 | 358.72 | 357.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 350.30 | 358.72 | 357.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 10:15:00 | 350.45 | 357.07 | 357.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 343.80 | 351.16 | 352.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 326.05 | 324.43 | 331.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 326.05 | 324.43 | 331.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 337.55 | 327.06 | 332.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 337.55 | 327.06 | 332.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 328.00 | 327.25 | 331.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 340.05 | 329.75 | 332.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 340.50 | 331.90 | 333.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 341.75 | 331.90 | 333.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 330.10 | 332.10 | 333.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:30:00 | 332.00 | 332.10 | 333.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 329.85 | 331.65 | 332.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 329.85 | 331.65 | 332.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 332.80 | 331.52 | 332.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 332.80 | 331.52 | 332.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 329.35 | 331.09 | 332.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 328.15 | 330.55 | 331.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 335.50 | 331.54 | 332.19 | SL hit (close>static) qty=1.00 sl=333.75 alert=retest2 |

### Cycle 25 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 334.15 | 332.70 | 332.65 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 331.05 | 332.37 | 332.50 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 335.60 | 333.02 | 332.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 346.55 | 335.72 | 334.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 338.90 | 343.60 | 341.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 338.90 | 343.60 | 341.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 338.90 | 343.60 | 341.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 337.25 | 343.60 | 341.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 341.60 | 343.20 | 341.41 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 337.75 | 340.64 | 340.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 315.90 | 335.11 | 338.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 326.90 | 325.82 | 330.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 326.90 | 325.82 | 330.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 329.80 | 326.93 | 329.88 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 337.05 | 331.87 | 331.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 343.05 | 335.16 | 333.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 10:15:00 | 336.80 | 337.56 | 335.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 10:15:00 | 336.80 | 337.56 | 335.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 336.80 | 337.56 | 335.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 340.95 | 335.46 | 335.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:30:00 | 338.80 | 336.55 | 335.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 14:15:00 | 334.10 | 335.86 | 335.65 | SL hit (close<static) qty=1.00 sl=334.20 alert=retest2 |

### Cycle 30 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 340.00 | 342.43 | 342.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 334.70 | 340.88 | 341.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 320.70 | 310.91 | 314.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 320.70 | 310.91 | 314.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 320.70 | 310.91 | 314.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 322.85 | 310.91 | 314.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 322.90 | 313.31 | 315.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 324.20 | 313.31 | 315.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 323.00 | 315.25 | 316.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 323.00 | 315.25 | 316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 332.70 | 318.74 | 317.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 342.70 | 325.27 | 320.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 327.25 | 336.36 | 331.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 327.25 | 336.36 | 331.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 327.25 | 336.36 | 331.26 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 326.80 | 328.97 | 329.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 322.50 | 327.24 | 328.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 315.45 | 314.06 | 318.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 315.45 | 314.06 | 318.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 315.45 | 314.06 | 318.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 320.50 | 314.06 | 318.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 320.75 | 315.40 | 318.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:45:00 | 323.00 | 315.40 | 318.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 11:15:00 | 320.30 | 316.38 | 318.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 12:15:00 | 320.20 | 316.38 | 318.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 319.90 | 317.66 | 318.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 15:15:00 | 320.00 | 317.66 | 318.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 15:15:00 | 320.00 | 318.13 | 318.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 301.50 | 318.13 | 318.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 316.55 | 312.61 | 312.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 316.55 | 312.61 | 312.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 321.85 | 315.51 | 313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 321.90 | 322.81 | 320.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 321.90 | 322.81 | 320.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 317.00 | 321.53 | 320.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 315.90 | 321.53 | 320.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 318.95 | 321.01 | 320.09 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 314.40 | 319.11 | 319.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 312.65 | 317.82 | 318.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 314.30 | 313.57 | 315.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 314.30 | 313.57 | 315.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 315.40 | 314.15 | 315.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 315.75 | 314.15 | 315.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 315.10 | 314.34 | 315.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:30:00 | 316.20 | 314.34 | 315.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 316.70 | 314.81 | 315.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 316.95 | 314.81 | 315.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 315.40 | 314.93 | 315.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 314.20 | 314.78 | 315.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 317.55 | 315.67 | 315.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 317.55 | 315.67 | 315.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 320.20 | 316.57 | 316.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 321.00 | 321.27 | 319.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:00:00 | 321.00 | 321.27 | 319.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 320.00 | 320.85 | 319.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 323.65 | 321.24 | 320.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 322.45 | 321.00 | 320.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 325.60 | 321.53 | 321.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 11:15:00 | 356.01 | 342.58 | 338.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 353.05 | 361.54 | 362.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 351.05 | 355.66 | 357.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 354.85 | 354.31 | 356.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 354.85 | 354.31 | 356.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 356.50 | 354.60 | 355.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 357.10 | 354.60 | 355.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 358.80 | 355.44 | 356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 357.60 | 355.44 | 356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 357.20 | 355.79 | 356.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 358.50 | 355.79 | 356.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 361.35 | 357.45 | 356.97 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 355.95 | 357.18 | 357.33 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 361.60 | 357.47 | 357.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 364.30 | 358.83 | 357.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 389.25 | 389.54 | 382.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 389.25 | 389.54 | 382.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 399.05 | 402.39 | 398.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 399.05 | 402.39 | 398.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 399.70 | 401.85 | 399.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 400.00 | 401.42 | 399.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 400.35 | 401.48 | 400.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:45:00 | 400.00 | 401.35 | 400.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 394.35 | 399.33 | 399.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 394.35 | 399.33 | 399.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 392.80 | 395.58 | 397.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 386.95 | 386.00 | 390.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:00:00 | 386.95 | 386.00 | 390.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 391.80 | 386.81 | 388.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 394.65 | 386.81 | 388.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 392.20 | 387.89 | 388.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 392.75 | 387.89 | 388.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 384.05 | 387.27 | 388.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 383.00 | 385.15 | 386.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 383.00 | 384.45 | 386.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 385.05 | 382.02 | 381.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 385.05 | 382.02 | 381.66 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 380.25 | 381.72 | 381.89 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 387.00 | 382.78 | 382.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 392.15 | 386.88 | 384.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 388.40 | 390.49 | 387.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 388.25 | 390.04 | 387.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 390.00 | 390.04 | 387.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 388.95 | 389.95 | 388.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 398.00 | 389.36 | 388.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 384.90 | 388.70 | 388.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 384.90 | 388.70 | 388.84 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 390.10 | 388.32 | 388.28 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 383.85 | 387.55 | 387.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 382.25 | 386.49 | 387.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 386.00 | 386.33 | 387.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 389.00 | 386.86 | 387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 389.00 | 386.86 | 387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 389.80 | 387.45 | 387.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 389.80 | 387.45 | 387.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 397.60 | 389.48 | 388.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 411.50 | 393.88 | 390.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 417.90 | 419.59 | 412.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 12:00:00 | 417.90 | 419.59 | 412.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 416.60 | 419.00 | 413.12 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 409.15 | 412.36 | 412.46 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 413.00 | 412.41 | 412.33 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 408.35 | 411.60 | 411.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 404.30 | 410.14 | 411.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 407.65 | 407.29 | 408.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 407.65 | 407.29 | 408.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 413.50 | 408.54 | 409.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 413.50 | 408.54 | 409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 413.80 | 409.59 | 409.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 409.85 | 409.59 | 409.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 09:15:00 | 389.36 | 401.12 | 403.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 402.60 | 395.60 | 398.02 | SL hit (close>ema200) qty=0.50 sl=395.60 alert=retest2 |

### Cycle 51 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 397.40 | 394.47 | 394.30 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 392.05 | 394.70 | 394.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 384.00 | 392.46 | 393.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 390.85 | 390.20 | 392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 394.35 | 391.03 | 392.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 393.55 | 391.03 | 392.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 395.35 | 391.89 | 392.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 395.35 | 391.89 | 392.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 398.30 | 393.17 | 393.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 403.70 | 395.28 | 393.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 394.55 | 396.61 | 395.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 393.00 | 395.89 | 395.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 393.70 | 395.89 | 395.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 394.00 | 395.51 | 395.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 402.10 | 395.51 | 395.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 09:15:00 | 442.31 | 419.91 | 412.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 420.55 | 425.14 | 425.34 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 432.50 | 425.48 | 425.33 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 422.45 | 425.74 | 425.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 15:15:00 | 420.00 | 424.59 | 425.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 420.30 | 414.79 | 418.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 419.40 | 415.71 | 418.43 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 429.10 | 421.68 | 420.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 430.40 | 424.32 | 422.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 10:15:00 | 424.20 | 424.30 | 422.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:30:00 | 425.60 | 424.30 | 422.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 423.25 | 424.47 | 422.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 423.25 | 424.47 | 422.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 425.70 | 424.72 | 423.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 423.20 | 424.72 | 423.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 430.20 | 427.80 | 425.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 425.60 | 427.80 | 425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 425.85 | 427.86 | 426.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 425.10 | 427.86 | 426.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 427.45 | 427.78 | 426.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:15:00 | 424.55 | 427.78 | 426.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 423.00 | 426.82 | 426.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 423.00 | 426.82 | 426.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 423.45 | 426.15 | 426.17 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 428.15 | 426.55 | 426.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 432.95 | 427.83 | 426.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 09:15:00 | 433.45 | 437.00 | 437.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 10:15:00 | 428.35 | 435.27 | 436.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 424.70 | 427.78 | 430.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 424.75 | 427.25 | 430.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 425.10 | 427.25 | 430.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:30:00 | 425.10 | 426.15 | 428.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 424.70 | 423.98 | 425.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 425.85 | 423.98 | 425.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 425.20 | 424.22 | 425.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 424.60 | 424.22 | 425.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 423.70 | 424.12 | 425.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 425.40 | 424.12 | 425.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 426.50 | 424.59 | 425.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 429.30 | 424.59 | 425.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 430.50 | 425.78 | 425.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 431.15 | 425.78 | 425.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 434.15 | 427.45 | 426.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 434.15 | 427.45 | 426.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 436.05 | 429.17 | 427.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 448.15 | 451.47 | 445.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:45:00 | 447.95 | 451.47 | 445.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 448.20 | 449.51 | 447.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 448.35 | 449.51 | 447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 448.20 | 449.14 | 447.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 445.70 | 449.14 | 447.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 446.60 | 448.63 | 447.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:45:00 | 446.40 | 448.63 | 447.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 447.30 | 448.36 | 447.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 449.00 | 448.36 | 447.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 440.75 | 447.14 | 447.09 | SL hit (close<static) qty=1.00 sl=446.60 alert=retest2 |

### Cycle 62 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 444.00 | 446.51 | 446.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 440.05 | 444.27 | 445.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 418.50 | 418.49 | 424.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 418.50 | 418.49 | 424.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 424.75 | 419.41 | 423.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 424.55 | 419.41 | 423.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 421.20 | 419.77 | 422.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 419.45 | 419.52 | 422.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 416.45 | 421.53 | 422.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 418.10 | 417.86 | 419.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 421.75 | 418.26 | 418.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 421.75 | 418.26 | 418.16 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 416.00 | 417.84 | 418.03 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 420.00 | 418.27 | 418.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 421.70 | 419.00 | 418.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 421.50 | 419.78 | 419.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 426.80 | 420.55 | 419.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 430.00 | 441.77 | 442.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 430.00 | 441.77 | 442.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 426.40 | 438.70 | 440.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 430.50 | 425.98 | 430.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 433.25 | 427.43 | 430.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 433.25 | 427.43 | 430.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 427.50 | 427.44 | 430.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:30:00 | 425.40 | 426.32 | 428.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 424.00 | 426.32 | 428.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 424.85 | 426.00 | 428.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 423.70 | 427.00 | 428.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 424.25 | 425.87 | 427.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 422.85 | 425.27 | 426.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 422.40 | 424.47 | 425.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 423.10 | 423.05 | 424.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 421.60 | 418.63 | 420.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 409.45 | 416.79 | 419.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 404.40 | 412.46 | 415.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 404.13 | 411.51 | 415.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 403.61 | 411.51 | 415.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 12:15:00 | 402.80 | 408.29 | 412.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 402.51 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 401.71 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 401.94 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 408.20 | 406.08 | 409.95 | SL hit (close>ema200) qty=0.50 sl=406.08 alert=retest2 |

### Cycle 67 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 395.25 | 391.76 | 391.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 399.55 | 394.27 | 392.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 13:15:00 | 395.45 | 395.96 | 394.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 395.45 | 395.96 | 394.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 393.05 | 395.38 | 394.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 393.05 | 395.38 | 394.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 393.45 | 394.99 | 394.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 396.80 | 394.99 | 394.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 388.85 | 396.33 | 396.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 388.85 | 396.33 | 396.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 387.15 | 393.23 | 394.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 390.55 | 389.00 | 391.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 09:30:00 | 390.00 | 389.00 | 391.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 389.85 | 389.17 | 391.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 390.60 | 389.17 | 391.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 405.70 | 392.22 | 392.21 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 391.45 | 393.01 | 393.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 388.50 | 391.82 | 392.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 382.90 | 381.51 | 384.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 382.90 | 381.51 | 384.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 386.95 | 382.60 | 385.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 386.90 | 382.60 | 385.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 386.95 | 383.47 | 385.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 387.25 | 383.47 | 385.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 388.80 | 386.40 | 386.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 391.40 | 387.62 | 386.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 14:15:00 | 401.75 | 402.39 | 399.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 15:00:00 | 401.75 | 402.39 | 399.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 399.80 | 401.75 | 399.56 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 394.60 | 398.69 | 398.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 392.30 | 396.70 | 397.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 397.70 | 390.86 | 393.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 392.35 | 391.16 | 393.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 395.20 | 391.16 | 393.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 393.80 | 391.69 | 393.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 393.50 | 391.69 | 393.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 394.95 | 392.34 | 393.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 394.95 | 392.34 | 393.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 395.55 | 392.98 | 393.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 397.30 | 392.98 | 393.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 397.80 | 393.95 | 394.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 397.80 | 393.95 | 394.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 397.70 | 394.70 | 394.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 403.75 | 396.51 | 395.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 393.90 | 398.89 | 397.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 398.00 | 398.71 | 397.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 398.90 | 398.71 | 397.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 402.00 | 408.23 | 408.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 402.00 | 408.23 | 408.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 399.00 | 406.38 | 407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 403.20 | 402.18 | 404.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:15:00 | 405.05 | 402.18 | 404.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 402.75 | 402.29 | 404.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 403.20 | 402.29 | 404.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 404.00 | 402.63 | 404.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 404.65 | 402.63 | 404.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 400.40 | 402.19 | 404.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:45:00 | 398.70 | 401.67 | 403.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 397.75 | 401.67 | 403.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 411.80 | 402.74 | 402.98 | SL hit (close>static) qty=1.00 sl=404.75 alert=retest2 |

### Cycle 75 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 408.90 | 403.97 | 403.52 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 402.00 | 405.26 | 405.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 401.00 | 404.41 | 405.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 405.15 | 403.45 | 404.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 403.55 | 403.47 | 404.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:30:00 | 403.20 | 403.56 | 404.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 403.10 | 403.52 | 404.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 400.70 | 403.52 | 404.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 402.75 | 395.25 | 397.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 397.95 | 395.79 | 397.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 397.00 | 396.17 | 397.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 383.04 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 382.94 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 382.61 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 387.80 | 386.47 | 388.55 | SL hit (close>ema200) qty=0.50 sl=386.47 alert=retest2 |

### Cycle 77 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 323.35 | 317.39 | 317.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 324.60 | 318.83 | 317.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 323.85 | 324.50 | 321.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 308.60 | 324.50 | 321.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 307.80 | 321.16 | 320.49 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 309.75 | 318.88 | 319.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 305.15 | 316.13 | 318.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 319.65 | 313.54 | 315.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 323.40 | 315.51 | 316.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 323.40 | 315.51 | 316.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 315.55 | 316.16 | 316.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 318.10 | 316.16 | 316.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 315.00 | 315.92 | 316.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 317.55 | 315.92 | 316.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 307.85 | 313.76 | 315.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 306.35 | 313.76 | 315.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 306.55 | 311.69 | 313.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 13:15:00 | 305.45 | 310.86 | 313.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 317.20 | 310.49 | 312.16 | SL hit (close>static) qty=1.00 sl=315.60 alert=retest2 |

### Cycle 79 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 320.80 | 313.74 | 313.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 327.10 | 322.00 | 320.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 344.60 | 346.33 | 337.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 344.60 | 346.33 | 337.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 333.25 | 341.79 | 339.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 332.00 | 341.79 | 339.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 329.10 | 339.25 | 338.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 329.10 | 339.25 | 338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 331.65 | 337.73 | 337.91 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 341.40 | 337.85 | 337.38 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 332.35 | 337.63 | 337.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 12:15:00 | 330.90 | 335.43 | 336.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 332.95 | 332.47 | 334.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:45:00 | 332.55 | 332.47 | 334.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 334.30 | 332.84 | 334.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 334.30 | 332.84 | 334.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 334.00 | 333.07 | 334.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 335.30 | 333.07 | 334.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 332.30 | 332.92 | 333.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 328.70 | 332.25 | 333.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 312.26 | 320.10 | 323.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 307.65 | 307.08 | 309.82 | SL hit (close>ema200) qty=0.50 sl=307.08 alert=retest2 |

### Cycle 83 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 299.00 | 296.70 | 296.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 300.45 | 297.45 | 296.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:45:00 | 297.60 | 298.20 | 297.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 299.35 | 299.55 | 298.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 296.30 | 298.05 | 298.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 296.30 | 298.05 | 298.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 14:15:00 | 294.30 | 297.30 | 297.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 12:15:00 | 280.80 | 283.31 | 286.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:45:00 | 280.80 | 282.91 | 285.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:00:00 | 280.35 | 282.31 | 284.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 290.05 | 282.90 | 283.41 | SL hit (close>static) qty=1.00 sl=287.75 alert=retest2 |

### Cycle 85 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 290.90 | 284.50 | 284.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 292.30 | 287.12 | 285.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 287.35 | 294.88 | 292.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 286.90 | 293.28 | 291.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 286.85 | 293.28 | 291.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 283.75 | 289.83 | 290.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 282.70 | 288.41 | 289.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 289.85 | 287.85 | 289.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 285.60 | 287.40 | 288.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 284.80 | 286.89 | 288.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 283.65 | 286.05 | 287.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 270.56 | 277.20 | 281.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 269.47 | 277.20 | 281.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 279.60 | 275.18 | 278.90 | SL hit (close>ema200) qty=0.50 sl=275.18 alert=retest2 |

### Cycle 87 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 275.50 | 267.39 | 267.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 281.80 | 272.07 | 270.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 270.20 | 275.95 | 273.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 275.30 | 275.82 | 274.10 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 272.00 | 273.34 | 273.38 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 275.65 | 273.80 | 273.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 11:15:00 | 280.30 | 275.58 | 274.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 274.50 | 275.86 | 274.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 275.25 | 275.74 | 274.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 276.00 | 275.74 | 274.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 267.00 | 274.03 | 274.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 267.00 | 274.03 | 274.23 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 276.60 | 272.14 | 271.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 281.90 | 278.79 | 277.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 278.70 | 279.43 | 278.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 279.40 | 279.43 | 278.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 282.20 | 279.98 | 278.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 284.35 | 280.90 | 279.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 14:15:00 | 286.35 | 287.95 | 288.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 286.35 | 287.95 | 288.16 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 290.75 | 288.40 | 288.32 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 286.50 | 288.02 | 288.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 11:15:00 | 285.45 | 287.51 | 287.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 288.75 | 287.79 | 288.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 288.00 | 287.83 | 288.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 286.50 | 287.60 | 287.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 272.18 | 274.99 | 278.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 15:15:00 | 271.95 | 270.74 | 274.49 | SL hit (close>ema200) qty=0.50 sl=270.74 alert=retest2 |

### Cycle 95 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 278.80 | 275.72 | 275.57 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 274.65 | 275.47 | 275.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 271.80 | 274.74 | 275.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 277.80 | 275.08 | 275.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 277.15 | 275.49 | 275.40 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 274.70 | 275.34 | 275.34 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 277.35 | 275.74 | 275.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 279.60 | 276.51 | 275.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 279.10 | 279.48 | 278.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 279.10 | 279.48 | 278.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 277.85 | 279.21 | 278.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 278.25 | 279.21 | 278.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 279.75 | 279.32 | 278.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:00:00 | 280.85 | 279.63 | 278.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 281.95 | 280.03 | 278.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:15:00 | 281.10 | 280.10 | 278.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 282.80 | 280.20 | 279.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-04 09:15:00 | 516.95 | 2024-12-05 11:15:00 | 568.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-20 09:15:00 | 599.45 | 2024-12-24 09:15:00 | 573.10 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-12-23 10:30:00 | 599.40 | 2024-12-24 09:15:00 | 573.10 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2024-12-30 13:00:00 | 553.70 | 2025-01-02 14:15:00 | 556.55 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-01-07 09:15:00 | 523.70 | 2025-01-08 09:15:00 | 497.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 09:45:00 | 525.00 | 2025-01-08 09:15:00 | 498.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 10:15:00 | 524.10 | 2025-01-08 09:15:00 | 497.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 09:15:00 | 523.70 | 2025-01-09 09:15:00 | 501.80 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-01-07 09:45:00 | 525.00 | 2025-01-09 09:15:00 | 501.80 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-01-07 10:15:00 | 524.10 | 2025-01-09 09:15:00 | 501.80 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-01-29 12:15:00 | 415.55 | 2025-01-29 14:15:00 | 429.00 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-01-29 13:30:00 | 415.45 | 2025-01-29 14:15:00 | 429.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-31 12:30:00 | 412.30 | 2025-02-01 12:15:00 | 430.75 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-02-01 11:30:00 | 413.30 | 2025-02-01 12:15:00 | 430.75 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2025-02-18 09:15:00 | 332.15 | 2025-02-18 15:15:00 | 347.35 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2025-02-21 12:15:00 | 370.00 | 2025-02-24 09:15:00 | 358.85 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-03-07 10:30:00 | 361.65 | 2025-03-11 09:15:00 | 351.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-03-10 10:00:00 | 362.00 | 2025-03-11 09:15:00 | 351.95 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-03-10 10:30:00 | 362.35 | 2025-03-11 09:15:00 | 351.95 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-03-10 11:30:00 | 362.20 | 2025-03-11 09:15:00 | 351.95 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-04-01 11:30:00 | 328.15 | 2025-04-01 12:15:00 | 335.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-04-15 09:15:00 | 340.95 | 2025-04-15 14:15:00 | 334.10 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-04-15 10:30:00 | 338.80 | 2025-04-15 14:15:00 | 334.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-04-16 12:15:00 | 338.70 | 2025-04-24 15:15:00 | 340.00 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-04-17 09:30:00 | 340.70 | 2025-04-24 15:15:00 | 340.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-04-23 13:45:00 | 347.30 | 2025-04-24 15:15:00 | 340.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-04-23 15:00:00 | 347.90 | 2025-04-24 15:15:00 | 340.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-05-13 09:15:00 | 301.50 | 2025-05-15 13:15:00 | 316.55 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-05-22 14:00:00 | 314.20 | 2025-05-23 10:15:00 | 317.55 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-27 11:45:00 | 323.65 | 2025-06-04 11:15:00 | 356.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 10:45:00 | 322.45 | 2025-06-04 11:15:00 | 354.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 09:15:00 | 325.60 | 2025-06-04 12:15:00 | 358.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 13:15:00 | 400.00 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-30 13:45:00 | 400.35 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-30 14:45:00 | 400.00 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-07 10:00:00 | 383.00 | 2025-07-10 10:15:00 | 385.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-07 11:30:00 | 383.00 | 2025-07-10 10:15:00 | 385.05 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-15 12:15:00 | 390.00 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-15 15:00:00 | 388.95 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-16 09:15:00 | 398.00 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-07-30 09:15:00 | 409.85 | 2025-08-01 09:15:00 | 389.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:15:00 | 409.85 | 2025-08-04 12:15:00 | 402.60 | STOP_HIT | 0.50 | 1.77% |
| BUY | retest2 | 2025-08-13 09:15:00 | 402.10 | 2025-08-21 09:15:00 | 442.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-11 11:45:00 | 424.70 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-09-11 13:30:00 | 424.75 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 14:15:00 | 425.10 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-09-12 10:30:00 | 425.10 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-09-23 14:15:00 | 449.00 | 2025-09-24 09:15:00 | 440.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-30 11:30:00 | 419.45 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-01 09:15:00 | 416.45 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-03 10:45:00 | 418.10 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-08 11:00:00 | 421.50 | 2025-10-17 11:15:00 | 430.00 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-10-09 09:15:00 | 426.80 | 2025-10-17 11:15:00 | 430.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-10-23 14:30:00 | 425.40 | 2025-11-03 09:15:00 | 404.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 15:00:00 | 424.00 | 2025-11-03 09:15:00 | 403.61 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-10-24 09:45:00 | 424.85 | 2025-11-03 12:15:00 | 402.80 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-10-24 14:15:00 | 423.70 | 2025-11-03 15:15:00 | 402.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 11:00:00 | 422.85 | 2025-11-03 15:15:00 | 401.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:00:00 | 422.40 | 2025-11-03 15:15:00 | 401.94 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-23 14:30:00 | 425.40 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-10-23 15:00:00 | 424.00 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-10-24 09:45:00 | 424.85 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-10-24 14:15:00 | 423.70 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2025-10-27 11:00:00 | 422.85 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-10-28 12:00:00 | 422.40 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-10-29 09:45:00 | 423.10 | 2025-11-07 09:15:00 | 401.28 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-10-31 09:45:00 | 421.60 | 2025-11-07 09:15:00 | 400.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 423.10 | 2025-11-10 14:15:00 | 380.16 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2025-11-03 09:15:00 | 404.40 | 2025-11-10 14:15:00 | 384.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 406.35 | 2025-11-10 14:15:00 | 386.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:15:00 | 408.10 | 2025-11-10 14:15:00 | 387.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 12:30:00 | 407.70 | 2025-11-10 14:15:00 | 387.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 421.60 | 2025-11-11 09:15:00 | 379.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-03 09:15:00 | 404.40 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-11-06 09:15:00 | 406.35 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-11-06 11:15:00 | 408.10 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-11-06 12:30:00 | 407.70 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-11-10 13:00:00 | 395.90 | 2025-11-17 12:15:00 | 395.25 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-11-12 10:30:00 | 396.00 | 2025-11-17 12:15:00 | 395.25 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-11-19 09:15:00 | 396.80 | 2025-11-21 09:15:00 | 388.85 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-11 09:15:00 | 398.90 | 2025-12-16 11:15:00 | 402.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-12-17 14:45:00 | 398.70 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-17 15:15:00 | 397.75 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-02 11:15:00 | 383.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-02 11:15:00 | 382.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-02 11:15:00 | 382.61 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-05 09:15:00 | 380.66 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-12-29 12:15:00 | 397.00 | 2026-01-05 13:15:00 | 377.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-06 09:15:00 | 360.63 | TARGET_HIT | 0.50 | 10.46% |
| SELL | retest2 | 2025-12-29 12:15:00 | 397.00 | 2026-01-06 13:15:00 | 357.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-02 10:15:00 | 306.35 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2026-02-02 12:15:00 | 306.55 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2026-02-02 13:15:00 | 305.45 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-02-19 10:45:00 | 328.70 | 2026-02-24 09:15:00 | 312.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:45:00 | 328.70 | 2026-02-27 10:15:00 | 307.65 | STOP_HIT | 0.50 | 6.40% |
| BUY | retest2 | 2026-03-09 11:45:00 | 297.60 | 2026-03-10 13:15:00 | 296.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-03-10 10:00:00 | 299.35 | 2026-03-10 13:15:00 | 296.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-13 12:15:00 | 280.80 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-13 13:45:00 | 280.80 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-16 10:00:00 | 280.35 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2026-03-20 12:45:00 | 284.80 | 2026-03-23 13:15:00 | 270.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 283.65 | 2026-03-23 13:15:00 | 269.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 284.80 | 2026-03-24 11:15:00 | 279.60 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2026-03-20 13:30:00 | 283.65 | 2026-03-24 11:15:00 | 279.60 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2026-03-25 10:00:00 | 284.20 | 2026-03-27 14:15:00 | 269.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:00:00 | 284.20 | 2026-04-01 09:15:00 | 269.35 | STOP_HIT | 0.50 | 5.23% |
| BUY | retest2 | 2026-04-10 15:15:00 | 276.00 | 2026-04-13 09:15:00 | 267.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-21 10:30:00 | 284.35 | 2026-04-24 14:15:00 | 286.35 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-04-27 14:30:00 | 286.50 | 2026-04-30 09:15:00 | 272.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 286.50 | 2026-04-30 15:15:00 | 271.95 | STOP_HIT | 0.50 | 5.08% |
