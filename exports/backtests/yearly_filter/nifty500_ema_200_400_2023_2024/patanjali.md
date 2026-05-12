# Patanjali Foods Ltd. (PATANJALI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 459.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 72 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 13 |
| TARGET_HIT | 9 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 54
- **Target hits / Stop hits / Partials:** 9 / 61 / 13
- **Avg / median % per leg:** 0.23% / -1.29%
- **Sum % (uncompounded):** 18.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 1 | 2.1% | 0 | 48 | 0 | -2.83% | -135.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 48 | 1 | 2.1% | 0 | 48 | 0 | -2.83% | -135.9% |
| SELL (all) | 35 | 28 | 80.0% | 9 | 13 | 13 | 4.42% | 154.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 28 | 80.0% | 9 | 13 | 13 | 4.42% | 154.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 83 | 29 | 34.9% | 9 | 61 | 13 | 0.23% | 18.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 457.28 | 515.66 | 515.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 444.08 | 503.12 | 509.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 15:15:00 | 468.50 | 468.35 | 483.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-22 09:15:00 | 477.43 | 468.35 | 483.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 483.33 | 468.80 | 483.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:45:00 | 484.67 | 468.80 | 483.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 13:15:00 | 485.62 | 468.96 | 483.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 14:00:00 | 485.62 | 468.96 | 483.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 487.35 | 469.15 | 483.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 487.35 | 469.15 | 483.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 487.77 | 469.33 | 483.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:15:00 | 487.37 | 469.33 | 483.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 488.80 | 482.57 | 487.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:30:00 | 490.87 | 482.57 | 487.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 487.00 | 482.62 | 487.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:30:00 | 486.68 | 482.65 | 487.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 14:15:00 | 486.67 | 482.69 | 487.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 486.82 | 482.75 | 487.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:15:00 | 484.67 | 482.61 | 487.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 482.98 | 482.61 | 487.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 481.97 | 482.61 | 487.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 479.02 | 482.09 | 486.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:15:00 | 462.35 | 481.21 | 486.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:15:00 | 462.34 | 481.21 | 486.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:15:00 | 462.48 | 481.21 | 486.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:15:00 | 460.44 | 480.99 | 486.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 12:15:00 | 457.87 | 480.76 | 486.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:15:00 | 455.07 | 480.51 | 485.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-18 09:15:00 | 471.67 | 471.47 | 479.67 | SL hit (close>ema200) qty=0.50 sl=471.47 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 510.65 | 479.90 | 479.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 512.68 | 481.36 | 480.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 516.87 | 518.12 | 503.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 10:00:00 | 516.87 | 518.12 | 503.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 589.07 | 616.43 | 590.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 589.07 | 616.43 | 590.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 586.37 | 616.13 | 590.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 584.27 | 616.13 | 590.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 579.67 | 582.74 | 580.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 573.68 | 582.74 | 580.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 571.78 | 582.63 | 580.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 571.78 | 582.63 | 580.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 573.02 | 582.53 | 580.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 573.65 | 582.35 | 579.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:30:00 | 573.97 | 582.21 | 579.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 573.33 | 582.08 | 579.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:45:00 | 573.22 | 581.89 | 579.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 566.63 | 581.46 | 579.64 | SL hit (close<static) qty=1.00 sl=570.47 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 586.70 | 594.63 | 594.65 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 605.88 | 594.69 | 594.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 610.65 | 594.85 | 594.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 602.18 | 602.43 | 598.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 14:00:00 | 602.18 | 602.43 | 598.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 601.08 | 602.42 | 599.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:45:00 | 602.92 | 602.42 | 599.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 598.63 | 602.38 | 599.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 592.32 | 602.38 | 599.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 595.08 | 602.31 | 598.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 593.23 | 602.31 | 598.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 595.07 | 602.23 | 598.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 595.07 | 602.23 | 598.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 599.28 | 601.63 | 598.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 12:30:00 | 606.50 | 601.63 | 598.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 604.95 | 606.04 | 602.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 586.17 | 605.82 | 601.99 | SL hit (close<static) qty=1.00 sl=597.20 alert=retest2 |

### Cycle 5 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 571.00 | 601.46 | 601.58 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 604.67 | 596.67 | 596.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 15:15:00 | 605.88 | 596.83 | 596.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 623.47 | 626.45 | 614.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 14:00:00 | 623.47 | 626.45 | 614.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 616.63 | 625.68 | 615.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 614.00 | 625.68 | 615.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 615.33 | 625.58 | 615.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 610.37 | 625.58 | 615.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 595.50 | 625.28 | 615.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 595.50 | 625.28 | 615.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 603.00 | 625.06 | 615.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:45:00 | 610.33 | 624.23 | 615.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 586.60 | 622.55 | 614.58 | SL hit (close<static) qty=1.00 sl=591.17 alert=retest2 |

### Cycle 7 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 572.37 | 608.86 | 608.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 554.83 | 594.44 | 600.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 556.17 | 556.13 | 569.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:30:00 | 556.67 | 556.13 | 569.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 568.33 | 556.28 | 568.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 568.33 | 556.28 | 568.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 571.63 | 556.43 | 568.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 571.63 | 556.43 | 568.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 570.23 | 556.57 | 568.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:15:00 | 568.30 | 556.57 | 568.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 580.63 | 556.94 | 568.37 | SL hit (close>static) qty=1.00 sl=572.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 650.53 | 578.28 | 578.05 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 584.00 | 595.72 | 595.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 579.00 | 594.57 | 595.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 595.85 | 593.00 | 594.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 596.85 | 593.03 | 594.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 593.05 | 593.12 | 594.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 590.80 | 593.14 | 594.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 591.95 | 592.18 | 593.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 598.45 | 592.18 | 593.61 | SL hit (close>static) qty=1.00 sl=598.35 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-02 12:30:00 | 486.68 | 2024-05-09 10:15:00 | 462.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 14:15:00 | 486.67 | 2024-05-09 10:15:00 | 462.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 09:15:00 | 486.82 | 2024-05-09 10:15:00 | 462.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 12:15:00 | 484.67 | 2024-05-09 11:15:00 | 460.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:15:00 | 481.97 | 2024-05-09 12:15:00 | 457.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 479.02 | 2024-05-09 13:15:00 | 455.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 12:30:00 | 486.68 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-05-02 14:15:00 | 486.67 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-05-03 09:15:00 | 486.82 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-05-06 12:15:00 | 484.67 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2024-05-06 13:15:00 | 481.97 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2024-05-08 09:15:00 | 479.02 | 2024-05-18 09:15:00 | 471.67 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2024-05-29 15:15:00 | 482.07 | 2024-06-03 09:15:00 | 495.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-05-30 14:45:00 | 480.90 | 2024-06-03 09:15:00 | 495.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-05-31 10:15:00 | 471.63 | 2024-06-03 09:15:00 | 495.00 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2024-06-04 09:15:00 | 462.67 | 2024-06-04 11:15:00 | 416.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 455.30 | 2024-06-04 11:15:00 | 409.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 12:30:00 | 573.65 | 2024-10-15 09:15:00 | 566.63 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-10-11 14:30:00 | 573.97 | 2024-10-15 09:15:00 | 566.63 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-10-14 10:00:00 | 573.33 | 2024-10-15 09:15:00 | 566.63 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-10-14 11:45:00 | 573.22 | 2024-10-15 09:15:00 | 566.63 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-16 15:00:00 | 589.33 | 2024-10-17 13:15:00 | 575.62 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-10-17 10:15:00 | 582.98 | 2024-10-17 13:15:00 | 575.62 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-10-17 11:15:00 | 584.05 | 2024-10-17 13:15:00 | 575.62 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-18 15:15:00 | 593.00 | 2024-10-23 10:15:00 | 578.33 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-10-22 14:30:00 | 585.07 | 2024-10-23 11:15:00 | 576.67 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-23 14:30:00 | 583.05 | 2024-10-25 09:15:00 | 571.55 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-10-28 14:45:00 | 586.92 | 2024-10-29 09:15:00 | 578.63 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-10-29 10:15:00 | 583.12 | 2024-11-21 14:15:00 | 587.32 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-11-21 12:15:00 | 595.15 | 2024-11-21 14:15:00 | 587.32 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-21 13:00:00 | 594.27 | 2024-11-25 11:15:00 | 587.07 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-11-22 14:45:00 | 598.00 | 2024-11-25 11:15:00 | 587.07 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-11-25 10:15:00 | 594.63 | 2024-12-20 14:15:00 | 586.68 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-11-29 09:15:00 | 603.33 | 2024-12-20 14:15:00 | 586.68 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-11-29 10:45:00 | 601.28 | 2024-12-20 14:15:00 | 586.68 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-12-13 11:45:00 | 600.18 | 2024-12-20 15:15:00 | 576.67 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2024-12-13 12:30:00 | 600.30 | 2024-12-20 15:15:00 | 576.67 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-12-20 09:30:00 | 609.63 | 2024-12-20 15:15:00 | 576.67 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2024-12-20 11:30:00 | 601.97 | 2024-12-20 15:15:00 | 576.67 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2024-12-20 12:15:00 | 601.92 | 2024-12-20 15:15:00 | 576.67 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-01-14 12:30:00 | 606.50 | 2025-01-27 10:15:00 | 586.17 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-01-24 14:30:00 | 604.95 | 2025-01-27 10:15:00 | 586.17 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-01-27 15:00:00 | 604.88 | 2025-01-28 09:15:00 | 587.70 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-01-28 14:30:00 | 601.00 | 2025-02-03 10:15:00 | 596.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-02-01 09:15:00 | 609.53 | 2025-02-03 10:15:00 | 596.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-02-01 10:45:00 | 610.35 | 2025-02-03 10:15:00 | 596.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-02-03 09:45:00 | 607.15 | 2025-02-03 10:15:00 | 596.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-03 10:15:00 | 607.60 | 2025-02-03 10:15:00 | 596.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-02-04 09:15:00 | 603.87 | 2025-02-05 11:15:00 | 595.48 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-02-04 14:45:00 | 600.40 | 2025-02-05 11:15:00 | 595.48 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-02-05 14:45:00 | 599.53 | 2025-02-06 12:15:00 | 595.97 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-02-06 14:00:00 | 599.63 | 2025-02-07 11:15:00 | 595.83 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-02-12 11:45:00 | 602.67 | 2025-02-12 15:15:00 | 589.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-02-12 12:45:00 | 603.95 | 2025-02-12 15:15:00 | 589.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-02-13 09:45:00 | 604.23 | 2025-02-17 09:15:00 | 586.53 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-02-14 14:45:00 | 602.65 | 2025-02-17 09:15:00 | 586.53 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-18 14:30:00 | 603.48 | 2025-02-28 09:15:00 | 553.82 | STOP_HIT | 1.00 | -8.23% |
| BUY | retest2 | 2025-02-19 09:15:00 | 603.67 | 2025-02-28 09:15:00 | 553.82 | STOP_HIT | 1.00 | -8.26% |
| BUY | retest2 | 2025-02-19 11:00:00 | 604.30 | 2025-02-28 09:15:00 | 553.82 | STOP_HIT | 1.00 | -8.35% |
| BUY | retest2 | 2025-02-19 11:45:00 | 603.33 | 2025-02-28 09:15:00 | 553.82 | STOP_HIT | 1.00 | -8.21% |
| BUY | retest2 | 2025-02-19 14:45:00 | 605.65 | 2025-02-28 09:15:00 | 553.82 | STOP_HIT | 1.00 | -8.56% |
| BUY | retest2 | 2025-05-07 14:45:00 | 610.33 | 2025-05-09 09:15:00 | 586.60 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-05-12 10:45:00 | 608.70 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-05-12 11:30:00 | 608.33 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-05-13 10:00:00 | 609.20 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-07-15 13:15:00 | 568.30 | 2025-07-15 14:15:00 | 580.63 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-10-23 13:45:00 | 593.05 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-24 09:15:00 | 590.80 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-28 10:00:00 | 591.95 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-03 09:15:00 | 583.55 | 2025-12-03 14:15:00 | 554.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 580.85 | 2025-12-03 14:15:00 | 551.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 580.50 | 2025-12-03 14:15:00 | 552.71 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-11-20 13:45:00 | 581.80 | 2025-12-03 14:15:00 | 552.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 581.80 | 2025-12-04 11:15:00 | 551.48 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-03 09:15:00 | 583.55 | 2025-12-04 15:15:00 | 525.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 580.85 | 2025-12-04 15:15:00 | 522.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 580.50 | 2025-12-04 15:15:00 | 522.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 13:45:00 | 581.80 | 2025-12-04 15:15:00 | 523.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 581.80 | 2025-12-04 15:15:00 | 523.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 565.60 | 2026-01-16 10:15:00 | 537.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 564.05 | 2026-01-16 12:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 565.60 | 2026-01-20 11:15:00 | 509.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 564.05 | 2026-01-20 12:15:00 | 507.64 | TARGET_HIT | 0.50 | 10.00% |
