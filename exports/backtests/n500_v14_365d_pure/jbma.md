# JBM Auto Ltd. (JBMA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 649.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 12 |
| TARGET_HIT | 10 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 17
- **Target hits / Stop hits / Partials:** 10 / 18 / 12
- **Avg / median % per leg:** 2.53% / 5.00%
- **Sum % (uncompounded):** 101.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.79% | -19.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.79% | -19.5% |
| SELL (all) | 33 | 23 | 69.7% | 10 | 11 | 12 | 3.66% | 120.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 23 | 69.7% | 10 | 11 | 12 | 3.66% | 120.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 23 | 57.5% | 10 | 18 | 12 | 2.53% | 101.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 643.20 | 679.68 | 679.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 640.00 | 673.70 | 676.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 658.30 | 654.93 | 663.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 658.30 | 654.93 | 663.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 670.15 | 655.08 | 663.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 672.70 | 655.08 | 663.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 671.60 | 655.25 | 663.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 668.40 | 655.25 | 663.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 660.50 | 655.58 | 664.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 660.00 | 655.58 | 664.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 13:00:00 | 660.15 | 655.65 | 663.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 627.00 | 653.51 | 661.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 627.14 | 653.51 | 661.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-08 14:15:00 | 594.00 | 641.66 | 653.67 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-08 14:15:00 | 594.13 | 641.66 | 653.67 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:30:00 | 650.95 | 633.75 | 645.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 14:15:00 | 618.40 | 633.15 | 644.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 626.70 | 624.10 | 636.50 | SL hit (close>ema200) qty=0.50 sl=624.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 754.35 | 644.98 | 644.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 765.85 | 666.12 | 656.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 14:15:00 | 678.10 | 678.33 | 664.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 678.10 | 678.33 | 664.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 666.35 | 677.59 | 666.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 664.75 | 677.59 | 666.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 661.80 | 677.43 | 666.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 661.80 | 677.43 | 666.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 662.70 | 677.28 | 666.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 661.20 | 677.28 | 666.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 661.50 | 674.83 | 665.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 661.65 | 674.83 | 665.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 663.00 | 674.48 | 665.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 665.00 | 674.48 | 665.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 653.85 | 674.04 | 665.67 | SL hit (close<static) qty=1.00 sl=661.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 666.55 | 662.84 | 661.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 664.30 | 665.04 | 662.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 665.90 | 665.04 | 662.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 658.70 | 665.00 | 662.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 658.70 | 665.00 | 662.80 | SL hit (close<static) qty=1.00 sl=661.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 658.70 | 665.00 | 662.80 | SL hit (close<static) qty=1.00 sl=661.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 658.70 | 665.00 | 662.80 | SL hit (close<static) qty=1.00 sl=661.05 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 658.70 | 665.00 | 662.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 659.20 | 664.94 | 662.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:30:00 | 658.80 | 664.94 | 662.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 652.15 | 664.55 | 662.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 653.35 | 664.55 | 662.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 632.00 | 660.78 | 660.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 630.60 | 659.97 | 660.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 631.30 | 593.02 | 614.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 642.05 | 593.51 | 614.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 642.05 | 593.51 | 614.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 641.00 | 598.82 | 615.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 644.05 | 598.82 | 615.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 643.60 | 599.27 | 615.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 643.60 | 599.27 | 615.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 631.60 | 618.75 | 622.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 635.35 | 618.75 | 622.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 606.95 | 617.25 | 621.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 601.00 | 616.74 | 621.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 601.00 | 616.28 | 620.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 601.20 | 616.28 | 620.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 602.25 | 616.14 | 620.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 572.14 | 611.05 | 617.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 570.95 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 570.95 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 571.14 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 540.90 | 601.36 | 611.95 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 540.90 | 601.36 | 611.95 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 541.08 | 601.36 | 611.95 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 542.02 | 601.36 | 611.95 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 567.35 | 583.51 | 599.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 561.85 | 586.19 | 595.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 566.30 | 585.99 | 595.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 567.05 | 585.80 | 595.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 566.40 | 585.60 | 595.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 533.76 | 576.79 | 588.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 537.98 | 576.79 | 588.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 538.70 | 576.79 | 588.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 538.08 | 576.79 | 588.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 509.67 | 573.27 | 586.41 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 510.34 | 573.27 | 586.41 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 509.76 | 573.27 | 586.41 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 12:15:00 | 505.67 | 571.38 | 585.26 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 562.70 | 539.54 | 562.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 562.70 | 539.54 | 562.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 556.70 | 539.71 | 562.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 560.60 | 539.71 | 562.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 565.95 | 539.97 | 562.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 565.95 | 539.97 | 562.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 571.15 | 540.28 | 562.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 561.45 | 541.50 | 563.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 561.65 | 541.70 | 563.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:45:00 | 559.95 | 541.88 | 563.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 554.45 | 542.34 | 562.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 556.10 | 542.48 | 562.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 559.35 | 542.48 | 562.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 554.70 | 542.60 | 562.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 551.30 | 542.60 | 562.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 580.70 | 543.15 | 562.79 | SL hit (close>static) qty=1.00 sl=578.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 580.70 | 543.15 | 562.79 | SL hit (close>static) qty=1.00 sl=578.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 580.70 | 543.15 | 562.79 | SL hit (close>static) qty=1.00 sl=578.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 580.70 | 543.15 | 562.79 | SL hit (close>static) qty=1.00 sl=578.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 580.70 | 543.15 | 562.79 | SL hit (close>static) qty=1.00 sl=562.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 548.40 | 543.48 | 562.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 575.05 | 543.75 | 562.14 | SL hit (close>static) qty=1.00 sl=562.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 10:00:00 | 551.20 | 547.67 | 562.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 13:15:00 | 523.64 | 546.68 | 561.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 573.55 | 546.35 | 561.23 | SL hit (close>ema200) qty=0.50 sl=546.35 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 543.60 | 547.53 | 561.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 567.15 | 547.58 | 561.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 567.15 | 547.58 | 561.07 | SL hit (close>static) qty=1.00 sl=562.95 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 567.15 | 547.58 | 561.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 562.65 | 547.73 | 561.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 552.55 | 547.83 | 561.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 556.40 | 547.97 | 561.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 579.50 | 549.87 | 561.16 | SL hit (close>static) qty=1.00 sl=568.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 579.50 | 549.87 | 561.16 | SL hit (close>static) qty=1.00 sl=568.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 12:15:00 | 619.60 | 570.37 | 570.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 628.25 | 572.52 | 571.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:45:00 | 706.25 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-06-16 14:45:00 | 724.80 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest2 | 2025-06-18 09:15:00 | 707.00 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-07-25 11:15:00 | 660.00 | 2025-07-31 09:15:00 | 627.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 13:00:00 | 660.15 | 2025-07-31 09:15:00 | 627.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 11:15:00 | 660.00 | 2025-08-08 14:15:00 | 594.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-25 13:00:00 | 660.15 | 2025-08-08 14:15:00 | 594.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-25 12:30:00 | 650.95 | 2025-08-26 14:15:00 | 618.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:30:00 | 650.95 | 2025-09-08 11:15:00 | 626.70 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest2 | 2025-10-10 14:15:00 | 665.00 | 2025-10-13 09:15:00 | 653.85 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-29 09:15:00 | 666.55 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-03 09:15:00 | 664.30 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-03 09:45:00 | 665.90 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-13 14:00:00 | 601.00 | 2026-01-20 12:15:00 | 572.14 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2026-01-14 10:30:00 | 601.00 | 2026-01-20 13:15:00 | 570.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 601.20 | 2026-01-20 13:15:00 | 570.95 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-01-14 12:00:00 | 602.25 | 2026-01-20 13:15:00 | 571.14 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-13 14:00:00 | 601.00 | 2026-01-23 10:15:00 | 540.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 601.00 | 2026-01-23 10:15:00 | 540.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 601.20 | 2026-01-23 10:15:00 | 541.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 12:00:00 | 602.25 | 2026-01-23 10:15:00 | 542.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 561.85 | 2026-03-02 09:15:00 | 533.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 10:00:00 | 566.30 | 2026-03-02 09:15:00 | 537.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 567.05 | 2026-03-02 09:15:00 | 538.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:30:00 | 566.40 | 2026-03-02 09:15:00 | 538.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 561.85 | 2026-03-04 09:15:00 | 509.67 | TARGET_HIT | 0.50 | 9.29% |
| SELL | retest2 | 2026-02-20 10:00:00 | 566.30 | 2026-03-04 09:15:00 | 510.34 | TARGET_HIT | 0.50 | 9.88% |
| SELL | retest2 | 2026-02-20 11:00:00 | 567.05 | 2026-03-04 09:15:00 | 509.76 | TARGET_HIT | 0.50 | 10.10% |
| SELL | retest2 | 2026-02-20 11:30:00 | 566.40 | 2026-03-04 12:15:00 | 505.67 | TARGET_HIT | 0.50 | 10.72% |
| SELL | retest2 | 2026-03-19 10:15:00 | 561.45 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-03-19 11:00:00 | 561.65 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2026-03-19 11:45:00 | 559.95 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2026-03-20 09:30:00 | 554.45 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-20 12:15:00 | 551.30 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2026-03-23 09:15:00 | 548.40 | 2026-03-24 09:15:00 | 575.05 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2026-03-27 10:00:00 | 551.20 | 2026-03-30 13:15:00 | 523.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:00:00 | 551.20 | 2026-04-01 09:15:00 | 573.55 | STOP_HIT | 0.50 | -4.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 543.60 | 2026-04-02 13:15:00 | 567.15 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2026-04-06 09:15:00 | 552.55 | 2026-04-08 09:15:00 | 579.50 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-04-06 11:00:00 | 556.40 | 2026-04-08 09:15:00 | 579.50 | STOP_HIT | 1.00 | -4.15% |
