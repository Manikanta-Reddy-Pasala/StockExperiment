# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 565.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 3 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 64 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 53
- **Target hits / Stop hits / Partials:** 9 / 63 / 19
- **Avg / median % per leg:** 0.83% / -1.16%
- **Sum % (uncompounded):** 75.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 8 | 17.8% | 0 | 41 | 4 | -1.45% | -65.5% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.19% | 25.5% |
| BUY @ 3rd Alert (retest2) | 37 | 0 | 0.0% | 0 | 37 | 0 | -2.46% | -90.9% |
| SELL (all) | 46 | 30 | 65.2% | 9 | 22 | 15 | 3.06% | 140.9% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.73% | 29.9% |
| SELL @ 3rd Alert (retest2) | 38 | 22 | 57.9% | 9 | 18 | 11 | 2.92% | 111.0% |
| retest1 (combined) | 16 | 16 | 100.0% | 0 | 8 | 8 | 3.46% | 55.4% |
| retest2 (combined) | 75 | 22 | 29.3% | 9 | 55 | 11 | 0.27% | 20.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 520.20 | 550.43 | 550.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 515.70 | 544.05 | 547.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 13:15:00 | 536.05 | 533.05 | 539.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-07 14:00:00 | 536.05 | 533.05 | 539.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 533.20 | 533.16 | 539.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 10:45:00 | 530.70 | 533.12 | 539.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 13:00:00 | 531.85 | 532.84 | 539.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 13:45:00 | 530.70 | 532.81 | 539.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-12 18:45:00 | 531.40 | 532.44 | 538.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 538.75 | 532.07 | 538.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:00:00 | 538.75 | 532.07 | 538.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 544.35 | 532.19 | 538.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-15 10:15:00 | 544.35 | 532.19 | 538.33 | SL hit (close>static) qty=1.00 sl=540.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 551.50 | 542.84 | 542.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 553.40 | 542.95 | 542.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 545.60 | 546.58 | 544.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 10:15:00 | 545.60 | 546.58 | 544.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 545.60 | 546.58 | 544.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 12:00:00 | 547.60 | 546.59 | 544.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:30:00 | 548.95 | 546.53 | 544.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 12:00:00 | 547.85 | 546.98 | 545.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 15:15:00 | 548.00 | 547.00 | 545.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 545.85 | 547.00 | 545.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:45:00 | 544.85 | 547.00 | 545.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 543.30 | 546.96 | 545.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 543.30 | 546.96 | 545.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 543.90 | 546.93 | 545.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 541.25 | 546.87 | 545.16 | SL hit (close<static) qty=1.00 sl=542.60 alert=retest2 |

### Cycle 3 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 518.80 | 543.84 | 543.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 512.65 | 539.74 | 541.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 14:15:00 | 534.85 | 534.16 | 538.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-29 15:00:00 | 534.85 | 534.16 | 538.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 537.10 | 534.00 | 537.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 536.35 | 534.00 | 537.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 537.95 | 534.04 | 537.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:45:00 | 539.25 | 534.04 | 537.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 11:15:00 | 538.60 | 534.08 | 537.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 11:30:00 | 538.90 | 534.08 | 537.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 538.50 | 534.12 | 537.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:15:00 | 538.60 | 534.12 | 537.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 539.05 | 534.17 | 537.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:45:00 | 539.50 | 534.17 | 537.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 538.60 | 534.24 | 537.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:30:00 | 538.00 | 534.24 | 537.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 536.70 | 534.27 | 537.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 535.40 | 536.73 | 538.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-18 09:15:00 | 481.86 | 533.12 | 536.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 15:15:00 | 553.55 | 524.10 | 524.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 09:15:00 | 554.95 | 524.41 | 524.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 590.00 | 594.60 | 572.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-19 09:30:00 | 581.35 | 594.60 | 572.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 578.75 | 592.93 | 573.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:30:00 | 582.65 | 585.22 | 573.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 11:15:00 | 587.10 | 583.22 | 573.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 14:15:00 | 582.55 | 582.95 | 573.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 588.40 | 582.84 | 573.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 580.35 | 584.77 | 575.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 582.25 | 584.77 | 575.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:30:00 | 583.00 | 585.29 | 576.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 13:15:00 | 574.10 | 584.62 | 577.00 | SL hit (close<static) qty=1.00 sl=574.45 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 10:15:00 | 566.75 | 572.53 | 572.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-06 11:15:00 | 562.85 | 572.43 | 572.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 15:15:00 | 571.95 | 571.89 | 572.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 15:15:00 | 571.95 | 571.89 | 572.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 571.95 | 571.89 | 572.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:15:00 | 574.05 | 571.89 | 572.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 574.55 | 571.92 | 572.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:15:00 | 577.20 | 571.92 | 572.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 579.25 | 571.99 | 572.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 578.50 | 571.99 | 572.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 579.15 | 572.59 | 572.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 14:15:00 | 582.30 | 572.91 | 572.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 623.10 | 624.73 | 607.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:00:00 | 623.10 | 624.73 | 607.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 729.80 | 754.35 | 729.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 729.80 | 754.35 | 729.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 729.70 | 754.10 | 729.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:15:00 | 733.70 | 754.10 | 729.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 743.00 | 753.99 | 729.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 15:15:00 | 747.00 | 753.99 | 729.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:45:00 | 743.90 | 751.42 | 731.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 752.80 | 751.19 | 731.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 11:45:00 | 746.05 | 751.02 | 732.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 734.90 | 750.11 | 732.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:45:00 | 732.70 | 750.11 | 732.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 733.25 | 749.83 | 732.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:45:00 | 739.50 | 749.72 | 732.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 10:00:00 | 738.45 | 748.93 | 733.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 13:15:00 | 740.25 | 749.74 | 735.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 738.45 | 749.05 | 737.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 735.45 | 748.92 | 737.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 735.90 | 748.92 | 737.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 717.95 | 748.61 | 737.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 717.95 | 748.61 | 737.52 | SL hit (close<static) qty=1.00 sl=728.65 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 694.05 | 729.45 | 729.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 09:15:00 | 685.95 | 727.62 | 728.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 676.15 | 674.04 | 689.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 676.15 | 674.04 | 689.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 596.00 | 569.11 | 593.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 595.35 | 569.11 | 593.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 594.85 | 569.36 | 593.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:30:00 | 597.80 | 569.36 | 593.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 597.60 | 571.28 | 593.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 594.70 | 571.54 | 593.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 12:15:00 | 595.85 | 571.54 | 593.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 594.05 | 572.08 | 593.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 595.40 | 572.71 | 593.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 590.35 | 572.88 | 593.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:30:00 | 589.40 | 573.03 | 593.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:30:00 | 588.65 | 573.47 | 593.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:45:00 | 589.20 | 573.76 | 593.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 589.35 | 574.64 | 593.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 14:15:00 | 564.97 | 574.83 | 592.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 14:15:00 | 566.06 | 574.83 | 592.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 14:15:00 | 564.35 | 574.83 | 592.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 14:15:00 | 565.63 | 574.83 | 592.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 559.93 | 574.25 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 559.22 | 574.25 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 559.74 | 574.25 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 559.88 | 574.25 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 535.23 | 571.46 | 588.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 613.30 | 591.81 | 591.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 617.20 | 593.07 | 592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 627.00 | 629.27 | 616.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 14:30:00 | 628.20 | 629.11 | 616.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 15:00:00 | 629.65 | 629.11 | 616.85 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 13:15:00 | 631.20 | 630.29 | 618.58 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 12:00:00 | 628.20 | 630.29 | 618.92 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 659.61 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 661.13 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 662.76 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 659.61 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 637.95 | 638.26 | 626.86 | SL hit (close<ema200) qty=0.50 sl=638.26 alert=retest1 |

### Cycle 9 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 615.00 | 630.43 | 630.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 612.35 | 629.36 | 629.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 628.90 | 625.95 | 628.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 624.45 | 625.93 | 628.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:30:00 | 623.05 | 625.87 | 627.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 621.50 | 625.83 | 627.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 629.90 | 625.82 | 627.91 | SL hit (close>static) qty=1.00 sl=629.30 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 630.35 | 608.33 | 608.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 631.00 | 614.74 | 612.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 660.50 | 660.98 | 645.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:15:00 | 662.70 | 660.98 | 645.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 645.85 | 660.36 | 645.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 647.85 | 660.36 | 645.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 645.00 | 660.21 | 645.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 641.85 | 660.21 | 645.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 639.65 | 659.85 | 645.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:45:00 | 639.85 | 659.85 | 645.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 649.70 | 659.03 | 645.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 655.30 | 650.87 | 644.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:45:00 | 654.40 | 652.12 | 645.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 653.10 | 652.10 | 645.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 654.65 | 651.98 | 645.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 642.50 | 651.76 | 645.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 642.50 | 651.76 | 645.70 | SL hit (close<static) qty=1.00 sl=643.15 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 600.60 | 644.37 | 644.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 596.40 | 638.86 | 641.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 565.10 | 564.36 | 591.77 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 12:00:00 | 562.45 | 564.36 | 591.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 13:00:00 | 561.05 | 564.32 | 591.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:45:00 | 561.75 | 564.23 | 590.63 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 12:15:00 | 563.00 | 564.23 | 590.50 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 534.33 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 533.00 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 533.66 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 534.85 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 548.20 | 545.59 | 569.85 | SL hit (close>ema200) qty=0.50 sl=545.59 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-08 10:45:00 | 530.70 | 2023-11-15 10:15:00 | 544.35 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2023-11-09 13:00:00 | 531.85 | 2023-11-15 10:15:00 | 544.35 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-11-09 13:45:00 | 530.70 | 2023-11-15 10:15:00 | 544.35 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2023-11-12 18:45:00 | 531.40 | 2023-11-15 10:15:00 | 544.35 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2023-12-05 12:00:00 | 547.60 | 2023-12-08 12:15:00 | 541.25 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-12-06 09:30:00 | 548.95 | 2023-12-08 12:15:00 | 541.25 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-12-07 12:00:00 | 547.85 | 2023-12-08 12:15:00 | 541.25 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2023-12-07 15:15:00 | 548.00 | 2023-12-08 12:15:00 | 541.25 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-12-12 09:15:00 | 546.15 | 2023-12-12 10:15:00 | 541.90 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-12-12 14:00:00 | 549.65 | 2023-12-13 10:15:00 | 541.60 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-12-12 14:45:00 | 548.00 | 2023-12-13 10:15:00 | 541.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-01-12 13:15:00 | 535.40 | 2024-01-18 09:15:00 | 481.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-27 11:30:00 | 534.35 | 2024-03-01 09:15:00 | 539.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-02-27 12:15:00 | 535.00 | 2024-03-01 09:15:00 | 539.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-02-27 12:45:00 | 531.85 | 2024-03-01 09:15:00 | 539.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-03 09:30:00 | 582.65 | 2024-05-22 13:15:00 | 574.10 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-08 11:15:00 | 587.10 | 2024-05-22 13:15:00 | 574.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-05-09 14:15:00 | 582.55 | 2024-05-27 14:15:00 | 573.15 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-05-10 09:15:00 | 588.40 | 2024-05-27 14:15:00 | 573.15 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-05-16 09:15:00 | 582.25 | 2024-05-27 14:15:00 | 573.15 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-05-21 11:30:00 | 583.00 | 2024-05-27 14:15:00 | 573.15 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-23 09:15:00 | 582.65 | 2024-05-29 09:15:00 | 569.75 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-05-23 11:15:00 | 582.55 | 2024-05-29 09:15:00 | 569.75 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-05-27 12:30:00 | 579.10 | 2024-05-29 13:15:00 | 555.75 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-05-27 13:45:00 | 578.05 | 2024-05-29 13:15:00 | 555.75 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-05-28 09:15:00 | 578.10 | 2024-05-29 13:15:00 | 555.75 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-05-28 12:45:00 | 578.30 | 2024-05-29 13:15:00 | 555.75 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2024-10-08 15:15:00 | 747.00 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-10-15 09:45:00 | 743.90 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2024-10-16 09:15:00 | 752.80 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2024-10-16 11:45:00 | 746.05 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-10-18 10:45:00 | 739.50 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-10-23 10:00:00 | 738.45 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-10-25 13:15:00 | 740.25 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-11-04 15:00:00 | 738.45 | 2024-11-05 09:15:00 | 717.95 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-03-25 11:30:00 | 594.70 | 2025-03-28 14:15:00 | 564.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 12:15:00 | 595.85 | 2025-03-28 14:15:00 | 566.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 14:15:00 | 594.05 | 2025-03-28 14:15:00 | 564.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 09:30:00 | 595.40 | 2025-03-28 14:15:00 | 565.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 11:30:00 | 589.40 | 2025-04-02 09:15:00 | 559.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 14:30:00 | 588.65 | 2025-04-02 09:15:00 | 559.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 09:45:00 | 589.20 | 2025-04-02 09:15:00 | 559.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 09:30:00 | 589.35 | 2025-04-02 09:15:00 | 559.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 11:30:00 | 594.70 | 2025-04-07 09:15:00 | 535.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 12:15:00 | 595.85 | 2025-04-07 09:15:00 | 536.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 14:15:00 | 594.05 | 2025-04-07 09:15:00 | 534.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 09:30:00 | 595.40 | 2025-04-07 09:15:00 | 535.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 11:30:00 | 589.40 | 2025-04-07 09:15:00 | 530.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 14:30:00 | 588.65 | 2025-04-07 09:15:00 | 529.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 09:45:00 | 589.20 | 2025-04-07 09:15:00 | 530.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 09:30:00 | 589.35 | 2025-04-07 09:15:00 | 530.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-16 13:15:00 | 582.25 | 2025-04-21 11:15:00 | 607.05 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-04-17 09:15:00 | 582.20 | 2025-04-21 11:15:00 | 607.05 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-05-08 14:45:00 | 582.50 | 2025-05-12 14:15:00 | 603.20 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2025-06-13 14:30:00 | 628.20 | 2025-07-01 09:15:00 | 659.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 15:00:00 | 629.65 | 2025-07-01 09:15:00 | 661.13 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-18 13:15:00 | 631.20 | 2025-07-01 09:15:00 | 662.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-19 12:00:00 | 628.20 | 2025-07-01 09:15:00 | 659.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 14:30:00 | 628.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2025-06-13 15:00:00 | 629.65 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-06-18 13:15:00 | 631.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2025-06-19 12:00:00 | 628.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest2 | 2025-07-24 09:15:00 | 631.35 | 2025-07-24 14:15:00 | 624.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-13 13:30:00 | 623.05 | 2025-08-14 09:15:00 | 629.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-13 15:00:00 | 621.50 | 2025-08-14 09:15:00 | 629.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-22 11:15:00 | 623.15 | 2025-08-25 12:15:00 | 629.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-26 09:15:00 | 622.30 | 2025-09-08 09:15:00 | 591.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 622.30 | 2025-09-22 09:15:00 | 612.80 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-09-04 11:30:00 | 609.05 | 2025-10-15 10:15:00 | 580.45 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-09-22 11:15:00 | 611.00 | 2025-10-15 10:15:00 | 580.16 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-09-04 11:30:00 | 609.05 | 2025-10-23 09:15:00 | 609.75 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2025-09-22 11:15:00 | 611.00 | 2025-10-23 09:15:00 | 609.75 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2025-09-22 14:30:00 | 610.70 | 2025-11-14 15:15:00 | 630.35 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-11-11 09:15:00 | 610.60 | 2025-11-14 15:15:00 | 630.35 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-02-03 09:15:00 | 655.30 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-06 09:45:00 | 654.40 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-06 11:15:00 | 653.10 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-09 10:30:00 | 654.65 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-02-19 10:30:00 | 655.10 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-02-19 13:00:00 | 651.55 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-19 14:45:00 | 651.25 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-02-20 09:30:00 | 653.25 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-03-02 11:00:00 | 656.95 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest1 | 2026-04-15 12:00:00 | 562.45 | 2026-04-23 09:15:00 | 534.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-15 13:00:00 | 561.05 | 2026-04-23 09:15:00 | 533.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-16 10:45:00 | 561.75 | 2026-04-23 09:15:00 | 533.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-16 12:15:00 | 563.00 | 2026-04-23 09:15:00 | 534.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-15 12:00:00 | 562.45 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2026-04-15 13:00:00 | 561.05 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest1 | 2026-04-16 10:45:00 | 561.75 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-04-16 12:15:00 | 563.00 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.63% |
