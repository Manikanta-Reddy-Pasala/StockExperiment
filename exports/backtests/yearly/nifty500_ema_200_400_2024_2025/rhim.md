# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5323 bars)
- **Last close:** 409.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 21 |
| TARGET_HIT | 15 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 19
- **Target hits / Stop hits / Partials:** 15 / 30 / 21
- **Avg / median % per leg:** 3.30% / 4.78%
- **Sum % (uncompounded):** 217.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 3 | 1 | 0 | 7.36% | 29.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 3 | 75.0% | 3 | 1 | 0 | 7.36% | 29.4% |
| SELL (all) | 62 | 44 | 71.0% | 12 | 29 | 21 | 3.04% | 188.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 44 | 71.0% | 12 | 29 | 21 | 3.04% | 188.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 47 | 71.2% | 15 | 30 | 21 | 3.30% | 217.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 669.80 | 630.50 | 630.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 670.95 | 634.46 | 632.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 14:15:00 | 655.65 | 657.70 | 646.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:30:00 | 652.95 | 657.70 | 646.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 654.90 | 666.75 | 655.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 655.00 | 666.75 | 655.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 654.95 | 666.63 | 655.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 654.65 | 666.63 | 655.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 649.05 | 665.98 | 655.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 650.65 | 665.98 | 655.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 648.35 | 664.62 | 655.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 648.35 | 664.62 | 655.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 649.40 | 658.29 | 653.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 649.40 | 658.29 | 653.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 645.80 | 658.17 | 653.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 645.80 | 658.17 | 653.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 610.60 | 648.55 | 648.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 608.10 | 647.45 | 648.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 11:15:00 | 619.60 | 617.85 | 629.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 619.60 | 617.85 | 629.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 640.50 | 604.61 | 617.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:15:00 | 658.80 | 604.61 | 617.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 622.55 | 607.34 | 618.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 622.55 | 607.34 | 618.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 627.90 | 616.21 | 621.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 627.90 | 616.21 | 621.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 620.00 | 616.35 | 621.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:45:00 | 620.05 | 616.35 | 621.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 623.95 | 616.45 | 621.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 620.00 | 616.45 | 621.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 615.00 | 616.43 | 621.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 612.50 | 616.43 | 621.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 613.60 | 615.64 | 620.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 582.92 | 611.96 | 617.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 12:15:00 | 581.88 | 611.15 | 617.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 600.00 | 599.53 | 608.63 | SL hit (close>ema200) qty=0.50 sl=599.53 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 482.30 | 462.39 | 462.29 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 445.00 | 462.58 | 462.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 442.65 | 462.38 | 462.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 457.15 | 455.25 | 458.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 12:15:00 | 457.15 | 455.25 | 458.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 457.15 | 455.25 | 458.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 13:00:00 | 457.15 | 455.25 | 458.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 464.15 | 454.90 | 458.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 464.75 | 454.90 | 458.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 463.35 | 454.98 | 458.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 463.35 | 454.98 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 457.85 | 455.09 | 458.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 457.85 | 455.09 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 458.60 | 455.13 | 458.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 458.60 | 455.13 | 458.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 459.45 | 455.17 | 458.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 460.40 | 455.17 | 458.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 462.45 | 455.24 | 458.19 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 471.40 | 460.41 | 460.38 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 456.05 | 460.34 | 460.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 460.13 | 460.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 456.70 | 456.67 | 458.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:00:00 | 456.70 | 456.67 | 458.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 456.60 | 456.25 | 458.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 457.20 | 456.25 | 458.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 456.90 | 456.18 | 457.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 456.90 | 456.18 | 457.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 460.80 | 456.14 | 457.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 460.80 | 456.14 | 457.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 461.45 | 456.20 | 457.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 461.00 | 456.20 | 457.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 491.95 | 456.61 | 458.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 491.95 | 456.61 | 458.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 510.00 | 459.86 | 459.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 521.80 | 461.77 | 460.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 476.00 | 476.30 | 469.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:30:00 | 475.50 | 476.30 | 469.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 468.20 | 476.19 | 469.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 468.20 | 476.19 | 469.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 467.00 | 476.10 | 469.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 465.45 | 476.10 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 466.65 | 476.01 | 469.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 466.45 | 476.01 | 469.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 466.45 | 475.50 | 469.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 466.45 | 475.50 | 469.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 469.85 | 475.45 | 469.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 472.00 | 475.09 | 469.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 471.00 | 475.39 | 470.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 470.85 | 474.68 | 470.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-22 10:15:00 | 518.10 | 477.88 | 473.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 470.75 | 484.50 | 484.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 467.50 | 481.60 | 482.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 493.05 | 466.94 | 474.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 501.50 | 467.28 | 474.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 480.50 | 468.00 | 474.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 15:15:00 | 456.47 | 467.97 | 474.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 460.60 | 457.40 | 465.65 | SL hit (close>ema200) qty=0.50 sl=457.40 alert=retest2 |

### Cycle 9 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 483.65 | 469.78 | 469.77 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 454.35 | 469.79 | 469.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 453.00 | 466.66 | 468.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 458.40 | 453.62 | 459.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 455.35 | 453.62 | 459.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 454.80 | 453.68 | 459.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 452.25 | 453.68 | 459.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 452.50 | 453.65 | 459.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 452.30 | 453.53 | 459.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:45:00 | 452.40 | 453.54 | 459.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 459.60 | 452.25 | 457.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 459.60 | 452.25 | 457.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 457.35 | 452.30 | 457.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 455.25 | 452.60 | 457.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 455.00 | 452.67 | 457.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 454.95 | 452.76 | 457.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 454.65 | 452.80 | 457.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 466.00 | 453.02 | 457.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 466.00 | 453.02 | 457.75 | SL hit (close>static) qty=1.00 sl=460.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 618.40 | 2024-05-13 15:15:00 | 625.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-05-13 12:30:00 | 620.95 | 2024-05-13 15:15:00 | 625.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-05-13 14:00:00 | 622.05 | 2024-05-13 15:15:00 | 625.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-08-29 10:15:00 | 612.50 | 2024-09-09 09:15:00 | 582.92 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2024-09-02 10:15:00 | 613.60 | 2024-09-09 12:15:00 | 581.88 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2024-08-29 10:15:00 | 612.50 | 2024-09-20 14:15:00 | 600.00 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2024-09-02 10:15:00 | 613.60 | 2024-09-20 14:15:00 | 600.00 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-09-24 11:30:00 | 614.20 | 2024-09-25 09:15:00 | 625.35 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-09-26 15:00:00 | 614.05 | 2024-10-08 09:15:00 | 583.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 15:00:00 | 614.05 | 2024-10-14 12:15:00 | 602.85 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2024-10-07 10:30:00 | 597.60 | 2024-10-15 10:15:00 | 635.95 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2024-10-09 09:30:00 | 600.75 | 2024-10-15 10:15:00 | 635.95 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-10-09 10:15:00 | 600.75 | 2024-10-15 10:15:00 | 635.95 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-10-10 11:00:00 | 600.25 | 2024-10-15 10:15:00 | 635.95 | STOP_HIT | 1.00 | -5.95% |
| SELL | retest2 | 2024-10-15 12:45:00 | 621.80 | 2024-10-18 09:15:00 | 594.98 | PARTIAL | 0.50 | 4.31% |
| SELL | retest2 | 2024-10-15 12:45:00 | 621.80 | 2024-10-18 09:15:00 | 606.85 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-10-15 14:15:00 | 626.30 | 2024-10-18 09:15:00 | 594.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 14:15:00 | 626.30 | 2024-10-18 09:15:00 | 606.85 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-10-15 15:00:00 | 626.30 | 2024-10-21 15:15:00 | 590.71 | PARTIAL | 0.50 | 5.68% |
| SELL | retest2 | 2024-10-17 09:15:00 | 621.85 | 2024-10-21 15:15:00 | 590.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:00:00 | 594.90 | 2024-10-25 09:15:00 | 565.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 15:00:00 | 626.30 | 2024-10-25 10:15:00 | 559.62 | TARGET_HIT | 0.50 | 10.65% |
| SELL | retest2 | 2024-10-17 09:15:00 | 621.85 | 2024-10-25 10:15:00 | 559.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 15:00:00 | 594.90 | 2024-10-31 14:15:00 | 593.90 | STOP_HIT | 0.50 | 0.17% |
| SELL | retest2 | 2024-11-04 09:15:00 | 583.25 | 2024-11-08 09:15:00 | 561.35 | PARTIAL | 0.50 | 3.75% |
| SELL | retest2 | 2024-11-07 09:15:00 | 590.90 | 2024-11-08 09:15:00 | 564.49 | PARTIAL | 0.50 | 4.47% |
| SELL | retest2 | 2024-11-07 15:00:00 | 594.20 | 2024-11-11 15:15:00 | 554.09 | PARTIAL | 0.50 | 6.75% |
| SELL | retest2 | 2024-11-04 09:15:00 | 583.25 | 2024-11-13 09:15:00 | 524.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 590.90 | 2024-11-13 09:15:00 | 531.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 15:00:00 | 594.20 | 2024-11-13 09:15:00 | 534.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-12 09:45:00 | 568.35 | 2024-12-19 09:15:00 | 539.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 11:15:00 | 566.50 | 2024-12-19 09:15:00 | 538.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:45:00 | 568.35 | 2024-12-20 14:15:00 | 511.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-18 11:15:00 | 566.50 | 2024-12-20 15:15:00 | 509.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-03 09:30:00 | 472.00 | 2025-07-22 10:15:00 | 518.10 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2025-07-07 14:45:00 | 471.00 | 2025-07-22 10:15:00 | 517.94 | TARGET_HIT | 1.00 | 9.96% |
| BUY | retest2 | 2025-07-09 10:00:00 | 470.85 | 2025-07-22 11:15:00 | 519.20 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2025-08-28 09:45:00 | 473.40 | 2025-09-10 12:15:00 | 470.75 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-07 15:15:00 | 480.50 | 2025-10-08 15:15:00 | 456.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 15:15:00 | 480.50 | 2025-10-29 09:15:00 | 460.60 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-11-14 09:45:00 | 485.20 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-14 12:15:00 | 486.10 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-11-18 09:15:00 | 485.90 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-12-22 10:15:00 | 452.25 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-12-22 11:45:00 | 452.50 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-12-23 09:30:00 | 452.30 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-12-23 11:45:00 | 452.40 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-01-01 10:00:00 | 455.25 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-01-01 11:30:00 | 455.00 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-01-02 09:15:00 | 454.95 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-01-02 11:30:00 | 454.65 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-01-08 15:15:00 | 462.60 | 2026-01-12 09:15:00 | 439.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 462.60 | 2026-01-14 11:15:00 | 459.10 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2026-01-14 13:15:00 | 461.55 | 2026-01-20 10:15:00 | 439.33 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2026-01-14 14:00:00 | 462.45 | 2026-01-20 12:15:00 | 438.47 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-14 13:15:00 | 461.55 | 2026-01-21 14:15:00 | 415.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 14:00:00 | 462.45 | 2026-01-21 14:15:00 | 416.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 14:00:00 | 460.00 | 2026-02-13 14:15:00 | 437.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 14:00:00 | 460.00 | 2026-02-16 09:15:00 | 463.50 | STOP_HIT | 0.50 | -0.76% |
| SELL | retest2 | 2026-02-16 11:30:00 | 458.60 | 2026-02-17 09:15:00 | 479.10 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-02-20 09:15:00 | 458.75 | 2026-02-24 12:15:00 | 436.81 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2026-02-20 11:00:00 | 459.80 | 2026-02-24 13:15:00 | 435.81 | PARTIAL | 0.50 | 5.22% |
| SELL | retest2 | 2026-02-20 11:30:00 | 458.00 | 2026-02-24 13:15:00 | 435.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 458.75 | 2026-03-02 12:15:00 | 413.82 | TARGET_HIT | 0.50 | 9.79% |
| SELL | retest2 | 2026-02-20 11:00:00 | 459.80 | 2026-03-02 13:15:00 | 412.88 | TARGET_HIT | 0.50 | 10.21% |
| SELL | retest2 | 2026-02-20 11:30:00 | 458.00 | 2026-03-02 13:15:00 | 412.20 | TARGET_HIT | 0.50 | 10.00% |
