# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1342.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 77 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 35 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 25
- **Target hits / Stop hits / Partials:** 7 / 32 / 11
- **Avg / median % per leg:** 1.59% / 0.06%
- **Sum % (uncompounded):** 79.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 11 | 36.7% | 5 | 21 | 4 | 0.83% | 25.0% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 22 | 3 | 13.6% | 1 | 21 | 0 | -1.59% | -35.0% |
| SELL (all) | 20 | 14 | 70.0% | 2 | 11 | 7 | 2.72% | 54.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 14 | 70.0% | 2 | 11 | 7 | 2.72% | 54.4% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 42 | 17 | 40.5% | 3 | 32 | 7 | 0.46% | 19.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 12:15:00 | 604.45 | 585.28 | 585.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 540.50 | 585.76 | 585.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 539.55 | 584.86 | 585.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 562.00 | 560.37 | 569.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-23 10:00:00 | 562.00 | 560.37 | 569.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 571.15 | 560.48 | 569.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:45:00 | 571.10 | 560.48 | 569.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 572.20 | 560.59 | 569.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:45:00 | 572.00 | 560.59 | 569.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 573.00 | 560.72 | 569.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:30:00 | 575.00 | 560.72 | 569.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 571.85 | 561.92 | 569.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:30:00 | 571.50 | 561.92 | 569.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 572.00 | 562.02 | 569.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 566.40 | 562.02 | 569.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 577.15 | 561.66 | 568.87 | SL hit (close>static) qty=1.00 sl=572.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 626.25 | 575.05 | 574.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 629.65 | 578.67 | 576.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 10:15:00 | 626.00 | 626.72 | 610.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 10:30:00 | 625.40 | 626.72 | 610.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 627.55 | 642.91 | 624.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:45:00 | 627.30 | 642.91 | 624.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 620.00 | 642.68 | 624.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:00:00 | 620.00 | 642.68 | 624.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 623.75 | 642.49 | 624.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:30:00 | 622.00 | 642.49 | 624.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 610.90 | 641.33 | 623.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:00:00 | 610.90 | 641.33 | 623.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 624.05 | 640.52 | 623.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 15:00:00 | 624.05 | 640.52 | 623.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 621.00 | 640.32 | 623.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:15:00 | 608.85 | 640.32 | 623.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 601.65 | 639.94 | 623.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:45:00 | 595.70 | 639.94 | 623.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 10:15:00 | 606.50 | 639.60 | 623.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-26 11:30:00 | 609.45 | 639.28 | 623.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-26 12:15:00 | 609.35 | 639.28 | 623.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 13:15:00 | 596.90 | 637.26 | 625.11 | SL hit (close<static) qty=1.00 sl=597.15 alert=retest2 |

### Cycle 4 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 602.55 | 619.07 | 619.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 14:15:00 | 600.75 | 618.89 | 619.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 624.80 | 618.66 | 618.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 10:15:00 | 624.80 | 618.66 | 618.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 624.80 | 618.66 | 618.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:00:00 | 624.80 | 618.66 | 618.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 628.75 | 618.76 | 618.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 12:00:00 | 628.75 | 618.76 | 618.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 632.00 | 619.20 | 619.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 632.80 | 620.32 | 619.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 11:15:00 | 667.05 | 667.47 | 652.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 11:45:00 | 668.60 | 667.47 | 652.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 654.10 | 667.07 | 652.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 656.45 | 667.07 | 652.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 653.60 | 666.94 | 652.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 649.95 | 666.94 | 652.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 651.70 | 666.66 | 652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 13:00:00 | 651.70 | 666.66 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 649.65 | 666.49 | 652.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 14:00:00 | 649.65 | 666.49 | 652.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 653.65 | 665.95 | 652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:00:00 | 653.65 | 665.95 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 652.40 | 665.82 | 652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:30:00 | 653.55 | 665.82 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 652.20 | 665.68 | 652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:30:00 | 652.15 | 665.68 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 652.00 | 665.55 | 652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:30:00 | 655.45 | 665.55 | 652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 654.45 | 665.44 | 652.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 657.25 | 665.31 | 652.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 10:15:00 | 648.30 | 664.46 | 652.39 | SL hit (close<static) qty=1.00 sl=649.95 alert=retest2 |

### Cycle 6 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 580.00 | 646.03 | 646.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 10:15:00 | 577.95 | 644.72 | 645.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 513.00 | 507.57 | 540.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-28 11:00:00 | 513.00 | 507.57 | 540.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 542.00 | 510.57 | 538.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:30:00 | 543.05 | 510.57 | 538.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 556.80 | 511.03 | 538.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:00:00 | 556.80 | 511.03 | 538.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 542.00 | 524.88 | 541.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:15:00 | 544.30 | 524.88 | 541.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 546.30 | 525.10 | 541.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:30:00 | 544.55 | 525.10 | 541.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 548.65 | 525.33 | 542.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 548.30 | 525.33 | 542.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 539.45 | 529.06 | 542.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:45:00 | 543.15 | 529.06 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 543.65 | 529.21 | 542.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:30:00 | 542.75 | 529.21 | 542.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 541.30 | 529.33 | 542.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:45:00 | 542.75 | 529.33 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 540.90 | 529.45 | 542.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:45:00 | 542.60 | 529.45 | 542.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 545.05 | 529.77 | 542.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 545.05 | 529.77 | 542.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 548.00 | 529.95 | 542.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:30:00 | 550.85 | 529.95 | 542.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 542.85 | 530.22 | 542.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:45:00 | 544.80 | 530.22 | 542.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 546.00 | 530.38 | 542.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:30:00 | 545.40 | 530.38 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 545.50 | 530.53 | 542.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 545.50 | 530.53 | 542.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 547.95 | 530.86 | 542.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 535.90 | 531.97 | 543.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 551.00 | 532.33 | 543.05 | SL hit (close>static) qty=1.00 sl=549.90 alert=retest2 |

### Cycle 7 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 612.00 | 551.34 | 551.23 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 545.50 | 553.82 | 553.86 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 574.50 | 554.03 | 553.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 580.00 | 555.53 | 554.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 545.60 | 556.63 | 555.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 531.50 | 556.38 | 555.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 531.50 | 556.38 | 555.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 553.95 | 554.66 | 554.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 559.05 | 554.66 | 554.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 13:15:00 | 614.96 | 572.48 | 564.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1155.80 | 1197.05 | 1197.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1145.00 | 1195.74 | 1196.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1200.65 | 1192.85 | 1195.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1204.30 | 1192.97 | 1195.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1220.10 | 1192.97 | 1195.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 1188.70 | 1192.88 | 1194.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 1190.35 | 1192.88 | 1194.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1189.55 | 1182.22 | 1189.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1189.55 | 1182.22 | 1189.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1193.15 | 1182.33 | 1189.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 1197.50 | 1182.33 | 1189.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1191.40 | 1182.42 | 1189.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 1194.25 | 1182.42 | 1189.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1190.85 | 1168.04 | 1179.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1190.85 | 1168.04 | 1179.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1171.20 | 1168.07 | 1179.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1154.00 | 1168.38 | 1179.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1096.30 | 1164.95 | 1177.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 1038.60 | 1159.11 | 1173.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1245.00 | 1122.37 | 1121.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 1252.40 | 1123.66 | 1122.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1221.20 | 1223.41 | 1184.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:00:00 | 1254.00 | 1223.71 | 1184.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:30:00 | 1250.90 | 1224.07 | 1185.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 12:00:00 | 1259.20 | 1224.07 | 1185.36 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 13:00:00 | 1259.30 | 1224.42 | 1185.73 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1316.70 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1313.45 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1322.16 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1322.27 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-15 09:15:00 | 1379.40 | 1245.36 | 1203.84 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 12 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 1394.40 | 1501.64 | 1501.80 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1564.00 | 1485.14 | 1485.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1575.70 | 1487.56 | 1486.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 1500.30 | 1500.23 | 1493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1499.30 | 1500.25 | 1493.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1499.30 | 1500.25 | 1493.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1493.00 | 1500.18 | 1493.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1479.80 | 1500.18 | 1493.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1485.00 | 1500.02 | 1493.53 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 1430.70 | 1487.87 | 1487.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1419.60 | 1481.72 | 1484.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 1531.60 | 1471.77 | 1478.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1522.00 | 1472.27 | 1479.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:30:00 | 1507.50 | 1475.06 | 1480.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 1501.00 | 1475.80 | 1480.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1505.50 | 1477.88 | 1481.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 1507.80 | 1478.65 | 1481.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1494.40 | 1479.04 | 1481.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1492.60 | 1479.04 | 1481.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1493.40 | 1479.18 | 1481.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 1493.20 | 1479.18 | 1481.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 1492.90 | 1479.45 | 1482.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1512.90 | 1479.92 | 1482.22 | SL hit (close>static) qty=1.00 sl=1502.40 alert=retest2 |

### Cycle 15 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1267.75 | 1073.09 | 1072.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 1270.05 | 1075.05 | 1073.23 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-25 09:15:00 | 566.40 | 2023-08-29 10:15:00 | 577.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-10-26 11:30:00 | 609.45 | 2023-11-02 13:15:00 | 596.90 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2023-10-26 12:15:00 | 609.35 | 2023-11-02 13:15:00 | 596.90 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2023-11-03 12:15:00 | 609.05 | 2023-11-08 10:15:00 | 622.10 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2023-11-06 09:30:00 | 616.50 | 2023-11-10 10:15:00 | 621.95 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2023-11-08 09:15:00 | 626.00 | 2023-11-10 10:15:00 | 621.95 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-08 11:45:00 | 626.30 | 2023-11-10 10:15:00 | 621.95 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-11-09 10:45:00 | 626.25 | 2023-11-28 13:15:00 | 602.55 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2023-11-09 11:15:00 | 626.85 | 2023-11-28 13:15:00 | 602.55 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2024-01-12 09:15:00 | 657.25 | 2024-01-15 10:15:00 | 648.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-01-16 11:30:00 | 659.50 | 2024-01-18 09:15:00 | 640.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-01-18 09:15:00 | 655.65 | 2024-01-18 09:15:00 | 640.60 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-01-19 09:30:00 | 660.30 | 2024-01-19 10:15:00 | 648.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-01-20 12:00:00 | 661.25 | 2024-01-20 14:15:00 | 647.05 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-04-19 09:15:00 | 535.90 | 2024-04-19 12:15:00 | 551.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-06-06 09:15:00 | 559.05 | 2024-06-18 13:15:00 | 614.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1154.00 | 2025-02-11 09:15:00 | 1096.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1154.00 | 2025-02-12 09:15:00 | 1038.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-11 09:30:00 | 1169.40 | 2025-04-11 10:15:00 | 1211.40 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest1 | 2025-05-07 11:00:00 | 1254.00 | 2025-05-12 10:15:00 | 1316.70 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 11:30:00 | 1250.90 | 2025-05-12 10:15:00 | 1313.45 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 12:00:00 | 1259.20 | 2025-05-12 10:15:00 | 1322.16 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 13:00:00 | 1259.30 | 2025-05-12 10:15:00 | 1322.27 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 11:00:00 | 1254.00 | 2025-05-15 09:15:00 | 1379.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 11:30:00 | 1250.90 | 2025-05-15 09:15:00 | 1375.99 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 12:00:00 | 1259.20 | 2025-05-15 09:15:00 | 1385.12 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 13:00:00 | 1259.30 | 2025-05-15 09:15:00 | 1385.23 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 11:00:00 | 1571.50 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2025-07-31 09:30:00 | 1576.30 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.88% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1555.00 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1575.70 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2025-08-22 10:00:00 | 1538.50 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-08-22 12:00:00 | 1547.50 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-22 14:30:00 | 1539.20 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-25 09:15:00 | 1539.60 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-29 10:30:00 | 1507.50 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-29 14:15:00 | 1501.00 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-31 11:15:00 | 1505.50 | 2025-11-04 14:15:00 | 1504.60 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-06 09:15:00 | 1432.12 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-06 09:15:00 | 1425.95 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-06 09:15:00 | 1430.22 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-06 09:15:00 | 1432.41 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-11 10:15:00 | 1366.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-11-24 15:15:00 | 1360.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-12-08 09:15:00 | 1288.89 | TARGET_HIT | 0.50 | 10.00% |
