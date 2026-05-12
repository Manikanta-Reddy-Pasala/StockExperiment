# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 572.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 4 |
| TARGET_HIT | 12 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 21 / 40
- **Target hits / Stop hits / Partials:** 12 / 45 / 4
- **Avg / median % per leg:** 0.79% / -1.07%
- **Sum % (uncompounded):** 48.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 13 | 33.3% | 10 | 29 | 0 | 0.66% | 25.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 39 | 13 | 33.3% | 10 | 29 | 0 | 0.66% | 25.6% |
| SELL (all) | 22 | 8 | 36.4% | 2 | 16 | 4 | 1.03% | 22.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 8 | 36.4% | 2 | 16 | 4 | 1.03% | 22.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 61 | 21 | 34.4% | 12 | 45 | 4 | 0.79% | 48.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 638.75 | 667.93 | 668.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 627.25 | 667.26 | 667.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 638.15 | 635.19 | 646.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-04 13:45:00 | 638.20 | 635.19 | 646.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 644.15 | 635.40 | 646.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:00:00 | 644.15 | 635.40 | 646.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 646.60 | 635.51 | 646.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 646.60 | 635.51 | 646.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 647.80 | 635.63 | 646.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:00:00 | 647.80 | 635.63 | 646.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 645.00 | 635.73 | 646.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:30:00 | 648.05 | 635.73 | 646.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 625.35 | 609.27 | 624.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 625.35 | 609.27 | 624.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 622.75 | 609.40 | 624.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 12:30:00 | 621.55 | 609.66 | 624.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 10:45:00 | 620.30 | 610.21 | 624.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 10:30:00 | 621.45 | 610.84 | 624.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 13:00:00 | 621.60 | 611.07 | 624.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 623.95 | 611.33 | 624.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 620.40 | 611.45 | 624.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 10:45:00 | 621.30 | 611.63 | 624.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 11:15:00 | 625.75 | 610.89 | 622.99 | SL hit (close>static) qty=1.00 sl=624.60 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 14:15:00 | 653.20 | 627.31 | 627.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 09:15:00 | 656.35 | 629.29 | 628.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 645.70 | 650.16 | 641.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 10:00:00 | 645.70 | 650.16 | 641.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 641.20 | 650.07 | 641.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 641.20 | 650.07 | 641.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 638.30 | 649.96 | 641.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:45:00 | 638.90 | 649.96 | 641.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 638.35 | 649.84 | 641.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 13:30:00 | 639.55 | 649.77 | 641.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 640.80 | 649.52 | 640.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 644.15 | 649.31 | 641.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 12:30:00 | 655.85 | 653.47 | 644.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-04 14:15:00 | 703.50 | 654.15 | 645.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 754.00 | 801.34 | 801.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 747.70 | 798.57 | 799.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 762.05 | 752.92 | 768.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 762.05 | 752.92 | 768.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 767.20 | 753.24 | 768.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:45:00 | 766.00 | 753.24 | 768.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 767.00 | 753.38 | 768.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:30:00 | 772.00 | 753.38 | 768.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 765.55 | 753.56 | 765.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:15:00 | 769.55 | 753.56 | 765.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 773.30 | 753.76 | 765.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 775.95 | 753.76 | 765.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 771.40 | 753.93 | 765.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:45:00 | 770.00 | 754.70 | 766.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:45:00 | 768.70 | 755.64 | 766.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:15:00 | 731.50 | 755.78 | 764.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:15:00 | 730.26 | 755.30 | 764.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-24 13:15:00 | 693.00 | 738.62 | 752.71 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 671.65 | 633.51 | 633.37 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 597.70 | 634.91 | 634.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 586.10 | 613.57 | 620.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 591.20 | 590.04 | 603.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 14:45:00 | 592.25 | 590.04 | 603.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 604.30 | 590.19 | 603.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 606.80 | 590.19 | 603.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 596.65 | 590.25 | 603.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 591.45 | 590.26 | 603.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 561.88 | 587.25 | 599.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 577.90 | 574.50 | 588.88 | SL hit (close>ema200) qty=0.50 sl=574.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 574.20 | 560.34 | 560.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 584.35 | 560.58 | 560.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 564.85 | 567.32 | 564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 558.45 | 567.23 | 564.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 558.45 | 567.23 | 564.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 559.10 | 567.15 | 564.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 567.10 | 566.83 | 564.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 561.35 | 566.65 | 564.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 557.95 | 566.50 | 563.98 | SL hit (close<static) qty=1.00 sl=558.05 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 540.35 | 561.90 | 561.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 539.25 | 561.06 | 561.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 561.75 | 559.02 | 560.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 556.05 | 558.99 | 560.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:30:00 | 557.00 | 558.99 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 562.35 | 558.96 | 560.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 562.35 | 558.96 | 560.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 560.00 | 558.97 | 560.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 535.45 | 558.97 | 560.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 557.95 | 550.15 | 554.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 571.45 | 550.72 | 554.86 | SL hit (close>static) qty=1.00 sl=565.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 610.55 | 558.89 | 558.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 635.55 | 575.72 | 568.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 576.30 | 585.05 | 575.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 577.00 | 584.97 | 575.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 584.45 | 584.24 | 575.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 585.00 | 584.26 | 575.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 584.60 | 584.26 | 575.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 13:30:00 | 584.85 | 584.21 | 575.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 579.75 | 584.04 | 575.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 585.25 | 584.01 | 575.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 573.95 | 584.08 | 575.91 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-03 12:30:00 | 621.55 | 2024-04-10 11:15:00 | 625.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-04-04 10:45:00 | 620.30 | 2024-04-10 11:15:00 | 625.75 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-04-05 10:30:00 | 621.45 | 2024-04-10 13:15:00 | 628.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-04-05 13:00:00 | 621.60 | 2024-04-10 13:15:00 | 628.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-04-08 09:15:00 | 620.40 | 2024-04-10 13:15:00 | 628.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-04-08 10:45:00 | 621.30 | 2024-04-10 13:15:00 | 628.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-04-12 12:00:00 | 621.20 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-04-23 12:15:00 | 621.20 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-04-23 13:15:00 | 617.95 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-04-23 15:00:00 | 616.10 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-04-24 11:00:00 | 618.55 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-04-24 15:00:00 | 618.80 | 2024-04-25 11:15:00 | 626.75 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-05-27 13:30:00 | 639.55 | 2024-06-04 14:15:00 | 703.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-28 09:15:00 | 640.80 | 2024-06-04 14:15:00 | 704.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-28 12:30:00 | 644.15 | 2024-06-04 14:15:00 | 708.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 655.85 | 2024-06-05 09:15:00 | 721.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 14:15:00 | 826.00 | 2024-08-21 11:15:00 | 781.20 | STOP_HIT | 1.00 | -5.42% |
| BUY | retest2 | 2024-08-14 09:30:00 | 813.35 | 2024-08-21 11:15:00 | 781.20 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-08-14 11:15:00 | 814.65 | 2024-08-21 11:15:00 | 781.20 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-08-14 14:15:00 | 814.00 | 2024-08-21 11:15:00 | 781.20 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-08-27 09:15:00 | 797.80 | 2024-10-07 11:15:00 | 799.50 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2024-08-27 13:15:00 | 804.60 | 2024-10-07 11:15:00 | 799.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-05 15:15:00 | 798.50 | 2024-10-07 11:15:00 | 799.50 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-09-06 13:30:00 | 798.00 | 2024-10-07 11:15:00 | 799.50 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-09-09 14:30:00 | 809.50 | 2024-10-07 12:15:00 | 793.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-09-11 10:15:00 | 806.90 | 2024-10-07 12:15:00 | 793.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-09-11 12:15:00 | 806.85 | 2024-10-07 12:15:00 | 793.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-09-11 15:15:00 | 807.15 | 2024-10-07 12:15:00 | 793.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-20 12:15:00 | 813.65 | 2024-10-11 13:15:00 | 784.50 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-09-23 09:15:00 | 812.00 | 2024-10-11 13:15:00 | 784.50 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-10-03 13:45:00 | 814.45 | 2024-10-11 13:15:00 | 784.50 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2024-10-04 09:30:00 | 815.75 | 2024-10-11 13:15:00 | 784.50 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2024-12-04 09:45:00 | 770.00 | 2024-12-12 10:15:00 | 731.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 09:45:00 | 768.70 | 2024-12-12 12:15:00 | 730.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 09:45:00 | 770.00 | 2024-12-24 13:15:00 | 693.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-05 09:45:00 | 768.70 | 2024-12-24 13:15:00 | 691.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 591.45 | 2025-07-29 10:15:00 | 561.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 591.45 | 2025-08-12 09:15:00 | 577.90 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-10-01 15:15:00 | 588.15 | 2025-10-15 15:15:00 | 558.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 15:15:00 | 588.15 | 2025-10-16 09:15:00 | 570.30 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2026-01-08 09:15:00 | 567.10 | 2026-01-09 11:15:00 | 557.95 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-09 10:15:00 | 561.35 | 2026-01-09 11:15:00 | 557.95 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-12 15:15:00 | 561.95 | 2026-01-13 09:15:00 | 556.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-27 09:15:00 | 535.45 | 2026-02-09 14:15:00 | 571.45 | STOP_HIT | 1.00 | -6.72% |
| SELL | retest2 | 2026-02-09 11:15:00 | 557.95 | 2026-02-09 14:15:00 | 571.45 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-03-11 09:15:00 | 584.45 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-03-11 10:45:00 | 585.00 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-03-11 11:30:00 | 584.60 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-03-11 13:30:00 | 584.85 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-12 12:15:00 | 585.25 | 2026-03-13 14:15:00 | 563.65 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-03-18 10:15:00 | 587.05 | 2026-03-25 09:15:00 | 645.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 10:00:00 | 585.80 | 2026-03-25 09:15:00 | 644.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:30:00 | 584.95 | 2026-03-25 09:15:00 | 643.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:30:00 | 600.00 | 2026-04-20 09:15:00 | 660.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 601.50 | 2026-04-20 09:15:00 | 654.55 | TARGET_HIT | 1.00 | 8.82% |
| BUY | retest2 | 2026-04-09 11:00:00 | 595.05 | 2026-04-20 09:15:00 | 657.14 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2026-04-09 11:30:00 | 597.40 | 2026-04-27 09:15:00 | 594.30 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-04-24 15:15:00 | 604.00 | 2026-04-28 13:15:00 | 592.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-04-27 10:30:00 | 604.65 | 2026-04-28 13:15:00 | 592.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-04-27 11:15:00 | 603.30 | 2026-05-04 09:15:00 | 556.45 | STOP_HIT | 1.00 | -7.77% |
| BUY | retest2 | 2026-04-30 12:15:00 | 605.50 | 2026-05-04 09:15:00 | 556.45 | STOP_HIT | 1.00 | -8.10% |
