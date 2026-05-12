# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 750.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 7 |
| TARGET_HIT | 9 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 55
- **Target hits / Stop hits / Partials:** 9 / 61 / 7
- **Avg / median % per leg:** -0.25% / -1.97%
- **Sum % (uncompounded):** -19.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 8 | 15.1% | 8 | 45 | 0 | -0.87% | -46.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 8 | 15.1% | 8 | 45 | 0 | -0.87% | -46.1% |
| SELL (all) | 24 | 14 | 58.3% | 1 | 16 | 7 | 1.11% | 26.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 14 | 58.3% | 1 | 16 | 7 | 1.11% | 26.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 77 | 22 | 28.6% | 9 | 61 | 7 | -0.25% | -19.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 12:15:00 | 307.50 | 292.53 | 292.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 13:15:00 | 312.95 | 292.73 | 292.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 09:15:00 | 310.60 | 312.29 | 305.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 304.25 | 312.12 | 305.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 304.25 | 312.12 | 305.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:00:00 | 304.25 | 312.12 | 305.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 301.85 | 312.02 | 305.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 301.85 | 312.02 | 305.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 299.10 | 303.09 | 302.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 15:00:00 | 299.10 | 303.09 | 302.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 300.60 | 303.00 | 302.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:00:00 | 300.60 | 303.00 | 302.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 302.00 | 302.98 | 302.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 13:15:00 | 301.60 | 302.98 | 302.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 301.85 | 302.97 | 302.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 302.90 | 302.95 | 302.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 15:15:00 | 300.80 | 302.95 | 302.34 | SL hit (close<static) qty=1.00 sl=301.45 alert=retest2 |

### Cycle 2 — SELL (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 14:15:00 | 419.80 | 426.55 | 426.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 418.55 | 426.02 | 426.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 417.90 | 413.56 | 418.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 417.90 | 413.56 | 418.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 417.90 | 413.56 | 418.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 411.00 | 421.27 | 421.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:00:00 | 415.60 | 421.01 | 421.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:45:00 | 412.95 | 420.92 | 421.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 425.35 | 420.89 | 421.65 | SL hit (close>static) qty=1.00 sl=424.30 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 457.40 | 422.44 | 422.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 460.60 | 423.48 | 422.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 657.85 | 663.02 | 613.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:00:00 | 657.85 | 663.02 | 613.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 612.70 | 662.18 | 613.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 598.50 | 662.18 | 613.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 591.50 | 661.47 | 613.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 591.50 | 661.47 | 613.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 569.30 | 660.56 | 613.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 570.10 | 660.56 | 613.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 599.00 | 596.73 | 591.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 599.70 | 596.73 | 591.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 590.50 | 596.70 | 591.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 585.25 | 596.70 | 591.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 589.30 | 596.62 | 591.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:00:00 | 600.45 | 587.93 | 587.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 602.00 | 588.42 | 587.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 606.95 | 588.42 | 587.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 599.05 | 589.53 | 588.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 589.55 | 591.38 | 589.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 595.30 | 591.38 | 589.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 591.60 | 591.89 | 589.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 581.75 | 591.75 | 589.78 | SL hit (close<static) qty=1.00 sl=584.60 alert=retest2 |

### Cycle 4 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 556.35 | 587.94 | 587.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 553.55 | 587.60 | 587.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 576.45 | 572.65 | 579.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 576.45 | 572.65 | 579.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 576.45 | 572.65 | 579.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 576.45 | 572.65 | 579.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 576.15 | 572.69 | 579.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:45:00 | 578.85 | 572.69 | 579.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 579.65 | 572.76 | 579.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 585.05 | 572.76 | 579.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 581.90 | 572.85 | 579.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 570.95 | 572.97 | 579.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:30:00 | 572.85 | 572.93 | 578.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 12:45:00 | 573.35 | 572.96 | 578.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 13:30:00 | 572.75 | 572.94 | 578.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 585.00 | 573.06 | 578.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 585.00 | 573.06 | 578.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 580.90 | 573.13 | 578.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 577.00 | 573.13 | 578.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 588.55 | 573.33 | 578.92 | SL hit (close>static) qty=1.00 sl=588.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 15:15:00 | 595.00 | 575.88 | 575.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 598.60 | 576.25 | 575.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 581.30 | 582.20 | 579.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 13:00:00 | 581.30 | 582.20 | 579.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 578.10 | 582.15 | 579.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 578.10 | 582.15 | 579.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 582.65 | 582.16 | 579.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:00:00 | 587.70 | 582.22 | 579.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 13:15:00 | 574.40 | 591.94 | 585.93 | SL hit (close<static) qty=1.00 sl=576.80 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 518.80 | 582.42 | 582.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 510.50 | 566.74 | 573.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 507.65 | 506.92 | 528.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 507.65 | 506.92 | 528.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 486.75 | 466.30 | 484.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 486.75 | 466.30 | 484.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 490.00 | 466.54 | 484.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:30:00 | 491.75 | 466.54 | 484.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 529.55 | 496.10 | 496.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 531.85 | 497.03 | 496.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 511.35 | 513.68 | 506.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 12:00:00 | 511.35 | 513.68 | 506.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 500.50 | 513.38 | 506.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 499.50 | 513.38 | 506.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 506.75 | 513.31 | 506.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 509.05 | 513.31 | 506.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 499.00 | 513.05 | 506.51 | SL hit (close<static) qty=1.00 sl=500.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 489.15 | 501.95 | 502.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 486.05 | 499.97 | 500.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 495.40 | 493.15 | 496.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 498.05 | 493.26 | 496.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 498.65 | 493.26 | 496.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 499.55 | 493.32 | 496.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 499.55 | 493.32 | 496.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 503.35 | 493.42 | 496.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 510.60 | 493.42 | 496.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 498.90 | 494.32 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 500.25 | 494.32 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 498.70 | 494.39 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 500.20 | 494.39 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 496.65 | 494.41 | 497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 498.60 | 494.41 | 497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 494.75 | 494.44 | 497.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 496.25 | 494.44 | 497.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 491.55 | 487.89 | 493.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 484.85 | 488.08 | 492.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:15:00 | 460.61 | 486.90 | 492.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-06 10:15:00 | 436.37 | 479.92 | 487.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 518.15 | 483.19 | 483.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 520.20 | 483.91 | 483.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 511.75 | 513.60 | 502.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 511.75 | 513.60 | 502.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 537.20 | 554.63 | 538.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 537.20 | 554.63 | 538.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 536.05 | 554.45 | 538.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 542.60 | 553.89 | 538.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:45:00 | 541.75 | 553.77 | 538.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 540.40 | 553.37 | 538.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 541.50 | 553.24 | 538.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 537.85 | 552.89 | 538.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 537.15 | 552.89 | 538.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 539.00 | 552.75 | 538.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 542.20 | 552.75 | 538.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 534.30 | 552.04 | 541.30 | SL hit (close<static) qty=1.00 sl=535.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-30 09:15:00 | 302.90 | 2023-08-30 15:15:00 | 300.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-09-01 11:30:00 | 302.70 | 2023-09-01 12:15:00 | 300.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-09-01 14:30:00 | 303.00 | 2023-09-14 14:15:00 | 333.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-04 09:15:00 | 303.60 | 2023-09-14 14:15:00 | 333.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-04 11:45:00 | 306.70 | 2023-09-15 09:15:00 | 337.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 11:00:00 | 411.00 | 2024-06-05 09:15:00 | 425.35 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-06-04 14:00:00 | 415.60 | 2024-06-05 09:15:00 | 425.35 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-06-04 14:45:00 | 412.95 | 2024-06-05 09:15:00 | 425.35 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-10-11 14:00:00 | 600.45 | 2024-10-22 10:15:00 | 581.75 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-10-14 10:30:00 | 602.00 | 2024-10-22 10:15:00 | 581.75 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-10-14 11:15:00 | 606.95 | 2024-10-22 10:15:00 | 581.75 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-10-15 10:30:00 | 599.05 | 2024-10-22 10:15:00 | 581.75 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-10-18 10:15:00 | 595.30 | 2024-10-22 11:15:00 | 576.40 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-10-22 09:15:00 | 591.60 | 2024-10-22 11:15:00 | 576.40 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-11-07 12:15:00 | 570.95 | 2024-11-11 10:15:00 | 588.55 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-11-08 11:30:00 | 572.85 | 2024-11-13 09:15:00 | 550.72 | PARTIAL | 0.50 | 3.86% |
| SELL | retest2 | 2024-11-08 12:45:00 | 573.35 | 2024-11-13 09:15:00 | 548.39 | PARTIAL | 0.50 | 4.35% |
| SELL | retest2 | 2024-11-08 13:30:00 | 572.75 | 2024-11-13 11:15:00 | 544.21 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-11-11 09:15:00 | 577.00 | 2024-11-13 11:15:00 | 544.68 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2024-11-11 12:00:00 | 579.70 | 2024-11-13 11:15:00 | 544.11 | PARTIAL | 0.50 | 6.14% |
| SELL | retest2 | 2024-11-11 13:30:00 | 577.25 | 2024-11-13 13:15:00 | 542.40 | PARTIAL | 0.50 | 6.04% |
| SELL | retest2 | 2024-11-08 11:30:00 | 572.85 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2024-11-08 12:45:00 | 573.35 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2024-11-08 13:30:00 | 572.75 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 0.92% |
| SELL | retest2 | 2024-11-11 09:15:00 | 577.00 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2024-11-11 12:00:00 | 579.70 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2024-11-11 13:30:00 | 577.25 | 2024-11-25 09:15:00 | 567.50 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-11-27 09:45:00 | 578.65 | 2024-12-02 09:15:00 | 590.90 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-12-03 14:30:00 | 547.60 | 2024-12-09 13:15:00 | 581.05 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2024-12-06 10:15:00 | 570.65 | 2024-12-09 13:15:00 | 581.05 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-12-06 12:15:00 | 573.65 | 2024-12-09 13:15:00 | 581.05 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-06 15:15:00 | 574.05 | 2024-12-09 13:15:00 | 581.05 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-13 10:15:00 | 560.10 | 2024-12-13 12:15:00 | 578.20 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-12-31 12:00:00 | 587.70 | 2025-01-13 13:15:00 | 574.40 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-01-14 11:00:00 | 587.60 | 2025-01-15 09:15:00 | 573.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-01-14 11:30:00 | 588.85 | 2025-01-15 09:15:00 | 573.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-01-14 13:00:00 | 588.40 | 2025-01-15 09:15:00 | 573.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-01-17 09:15:00 | 582.65 | 2025-01-22 09:15:00 | 578.55 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-17 12:45:00 | 582.60 | 2025-01-22 09:15:00 | 578.55 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-17 14:45:00 | 581.55 | 2025-01-22 09:15:00 | 578.55 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-01-20 11:00:00 | 583.80 | 2025-01-22 09:15:00 | 578.55 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-01-20 12:15:00 | 594.00 | 2025-01-24 09:15:00 | 582.40 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-21 12:00:00 | 598.30 | 2025-01-24 12:15:00 | 581.20 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-01-21 12:45:00 | 595.75 | 2025-01-24 12:15:00 | 581.20 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-01-21 14:00:00 | 595.30 | 2025-01-27 09:15:00 | 554.70 | STOP_HIT | 1.00 | -6.82% |
| BUY | retest2 | 2025-01-23 09:15:00 | 593.25 | 2025-01-27 09:15:00 | 554.70 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest2 | 2025-01-24 10:45:00 | 591.05 | 2025-01-27 09:15:00 | 554.70 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest2 | 2025-01-24 11:15:00 | 599.50 | 2025-01-27 09:15:00 | 554.70 | STOP_HIT | 1.00 | -7.47% |
| BUY | retest2 | 2025-06-16 11:15:00 | 509.05 | 2025-06-17 12:15:00 | 499.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-31 09:15:00 | 484.85 | 2025-08-01 10:15:00 | 460.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 484.85 | 2025-08-06 10:15:00 | 436.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-11-12 09:15:00 | 542.60 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-11-12 09:45:00 | 541.75 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-12 12:45:00 | 540.40 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-12 13:45:00 | 541.50 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-13 11:15:00 | 542.20 | 2025-11-24 10:15:00 | 534.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-25 11:15:00 | 541.75 | 2025-12-19 09:15:00 | 595.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-25 12:30:00 | 540.95 | 2025-12-19 09:15:00 | 595.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-08 14:45:00 | 542.55 | 2025-12-19 09:15:00 | 596.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-19 14:15:00 | 581.45 | 2026-01-20 10:15:00 | 564.10 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-01-23 09:45:00 | 581.90 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-01-23 10:15:00 | 584.25 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-01-23 12:15:00 | 581.45 | 2026-01-23 14:15:00 | 564.35 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2026-01-27 09:45:00 | 583.40 | 2026-01-27 12:15:00 | 573.45 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-02-04 09:15:00 | 587.05 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2026-02-04 12:15:00 | 584.35 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-02-04 15:15:00 | 583.95 | 2026-02-06 09:15:00 | 567.25 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2026-02-20 13:30:00 | 585.15 | 2026-03-02 12:15:00 | 569.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-06 12:30:00 | 585.35 | 2026-03-09 09:15:00 | 558.85 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2026-03-11 09:15:00 | 591.25 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-03-11 13:15:00 | 586.85 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-03-12 11:00:00 | 588.15 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-12 15:00:00 | 588.65 | 2026-03-13 09:15:00 | 568.65 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-18 14:45:00 | 586.60 | 2026-03-19 12:15:00 | 570.45 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-20 09:15:00 | 588.15 | 2026-03-23 11:15:00 | 569.30 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-03-23 13:15:00 | 571.15 | 2026-03-25 11:15:00 | 628.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 13:45:00 | 572.00 | 2026-03-25 11:15:00 | 629.20 | TARGET_HIT | 1.00 | 10.00% |
