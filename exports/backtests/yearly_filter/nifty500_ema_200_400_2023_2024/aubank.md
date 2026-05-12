# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1051.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT2_SKIP | 2 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 61 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 56
- **Target hits / Stop hits / Partials:** 2 / 63 / 6
- **Avg / median % per leg:** -1.06% / -1.70%
- **Sum % (uncompounded):** -75.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 7 | 17.5% | 0 | 38 | 2 | -1.58% | -63.4% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.69% | 4.1% |
| BUY @ 3rd Alert (retest2) | 34 | 4 | 11.8% | 0 | 34 | 0 | -1.99% | -67.5% |
| SELL (all) | 31 | 8 | 25.8% | 2 | 25 | 4 | -0.38% | -11.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 8 | 25.8% | 2 | 25 | 4 | -0.38% | -11.9% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.69% | 4.1% |
| retest2 (combined) | 65 | 12 | 18.5% | 2 | 59 | 4 | -1.22% | -79.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:15:00 | 755.00 | 749.62 | 717.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:30:00 | 751.60 | 748.14 | 719.63 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:15:00 | 752.75 | 748.12 | 720.46 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 14:15:00 | 756.00 | 748.03 | 721.10 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 14:15:00 | 789.18 | 753.26 | 728.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 14:15:00 | 790.39 | 753.26 | 728.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-07-10 11:15:00 | 752.75 | 754.74 | 730.41 | SL hit (close<ema200) qty=0.50 sl=754.74 alert=retest1 |

### Cycle 2 — SELL (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 14:15:00 | 700.60 | 732.08 | 732.11 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 10:15:00 | 749.70 | 730.07 | 730.03 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 704.60 | 730.51 | 730.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 689.40 | 715.64 | 721.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 11:15:00 | 697.40 | 693.85 | 706.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 11:45:00 | 696.70 | 693.85 | 706.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 710.50 | 694.09 | 706.26 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 742.90 | 714.16 | 714.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 13:15:00 | 746.70 | 714.49 | 714.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 730.95 | 733.78 | 726.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 15:00:00 | 730.95 | 733.78 | 726.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 755.75 | 767.47 | 752.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:30:00 | 753.05 | 767.47 | 752.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 747.50 | 767.27 | 752.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 12:00:00 | 747.50 | 767.27 | 752.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 755.00 | 767.15 | 752.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 13:15:00 | 757.80 | 767.15 | 752.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 14:00:00 | 757.20 | 767.05 | 752.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 15:00:00 | 759.45 | 766.97 | 752.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 12:15:00 | 737.80 | 766.08 | 752.03 | SL hit (close<static) qty=1.00 sl=746.75 alert=retest2 |

### Cycle 6 — SELL (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 11:15:00 | 621.55 | 742.13 | 742.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 606.40 | 697.81 | 717.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 589.55 | 584.87 | 618.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 15:00:00 | 589.55 | 584.87 | 618.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 634.50 | 586.08 | 616.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:00:00 | 634.50 | 586.08 | 616.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 628.00 | 589.28 | 616.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:15:00 | 626.95 | 589.28 | 616.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:45:00 | 627.00 | 589.64 | 616.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 13:15:00 | 634.40 | 590.43 | 616.86 | SL hit (close>static) qty=1.00 sl=632.30 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 649.60 | 623.60 | 623.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 653.20 | 626.48 | 625.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 618.05 | 628.54 | 626.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 618.05 | 628.54 | 626.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 618.05 | 628.54 | 626.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 618.05 | 628.54 | 626.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 630.25 | 628.56 | 626.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 621.30 | 628.56 | 626.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 630.40 | 628.58 | 626.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:30:00 | 628.75 | 628.58 | 626.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 629.90 | 628.59 | 626.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 629.90 | 628.59 | 626.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 628.00 | 628.59 | 626.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 10:30:00 | 639.80 | 628.86 | 626.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 638.10 | 655.93 | 648.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:00:00 | 637.20 | 653.39 | 647.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:00:00 | 638.40 | 650.28 | 646.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 646.85 | 650.22 | 646.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 646.85 | 650.22 | 646.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 646.85 | 650.19 | 646.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 646.85 | 650.19 | 646.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 649.15 | 650.18 | 646.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 14:15:00 | 655.60 | 650.18 | 646.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 652.10 | 650.29 | 646.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 652.00 | 650.38 | 646.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 13:15:00 | 652.50 | 650.38 | 646.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 649.00 | 650.85 | 647.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 645.55 | 650.85 | 647.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 647.65 | 650.82 | 647.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-25 12:15:00 | 642.65 | 650.71 | 647.22 | SL hit (close<static) qty=1.00 sl=646.45 alert=retest2 |

### Cycle 8 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 625.35 | 645.12 | 645.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 607.70 | 643.34 | 644.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 635.00 | 632.31 | 637.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:45:00 | 636.80 | 632.31 | 637.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 632.55 | 632.07 | 637.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 628.30 | 632.46 | 637.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 649.00 | 632.61 | 637.02 | SL hit (close>static) qty=1.00 sl=639.05 alert=retest2 |

### Cycle 9 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 683.30 | 641.21 | 641.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 12:15:00 | 691.90 | 643.49 | 642.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 13:15:00 | 707.65 | 712.95 | 690.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 14:00:00 | 707.65 | 712.95 | 690.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 692.10 | 711.96 | 691.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 691.20 | 711.96 | 691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 691.00 | 711.75 | 691.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 691.00 | 711.75 | 691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 689.90 | 711.54 | 691.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:30:00 | 689.65 | 711.54 | 691.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 690.55 | 711.33 | 691.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 690.55 | 711.33 | 691.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 693.00 | 711.14 | 691.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 686.95 | 711.14 | 691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 684.90 | 710.88 | 691.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 684.90 | 710.88 | 691.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 689.95 | 710.67 | 691.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:30:00 | 692.90 | 710.14 | 691.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 692.80 | 709.14 | 691.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 692.55 | 708.29 | 691.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 693.10 | 707.97 | 691.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 697.95 | 707.87 | 691.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 679.05 | 705.88 | 691.45 | SL hit (close<static) qty=1.00 sl=682.90 alert=retest2 |

### Cycle 10 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 619.75 | 680.33 | 680.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 615.90 | 679.11 | 679.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 587.75 | 577.61 | 601.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-30 15:00:00 | 587.75 | 577.61 | 601.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 601.05 | 573.42 | 590.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 601.00 | 573.42 | 590.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 604.80 | 573.73 | 590.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:00:00 | 604.80 | 573.73 | 590.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 596.35 | 582.41 | 592.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 591.50 | 582.47 | 592.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 15:15:00 | 588.00 | 582.56 | 592.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 561.92 | 582.36 | 592.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 558.60 | 582.36 | 592.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 581.55 | 580.92 | 590.97 | SL hit (close>ema200) qty=0.50 sl=580.92 alert=retest2 |

### Cycle 11 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 616.00 | 560.05 | 559.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 09:15:00 | 647.10 | 560.92 | 560.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 790.55 | 791.71 | 748.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 791.35 | 791.71 | 748.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 747.40 | 791.38 | 751.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 738.30 | 791.38 | 751.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 754.90 | 791.01 | 751.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 758.00 | 759.03 | 746.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 744.60 | 758.68 | 746.85 | SL hit (close<static) qty=1.00 sl=745.05 alert=retest2 |

### Cycle 12 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 703.30 | 742.58 | 742.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 700.30 | 742.16 | 742.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 729.45 | 722.31 | 729.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:00:00 | 729.45 | 722.31 | 729.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 732.05 | 722.41 | 729.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 732.05 | 722.41 | 729.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 728.85 | 722.48 | 729.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 725.35 | 728.13 | 731.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 727.00 | 728.14 | 731.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 727.15 | 728.14 | 731.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 723.95 | 728.10 | 731.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 735.35 | 728.18 | 731.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 735.35 | 728.18 | 731.59 | SL hit (close>static) qty=1.00 sl=733.50 alert=retest2 |

### Cycle 13 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 768.20 | 734.57 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 772.40 | 742.13 | 738.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 977.20 | 978.09 | 937.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:45:00 | 978.05 | 978.09 | 937.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 950.05 | 984.00 | 950.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 950.05 | 984.00 | 950.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 950.55 | 983.67 | 950.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 950.70 | 983.67 | 950.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 946.95 | 983.30 | 950.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:00:00 | 946.95 | 983.30 | 950.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 962.65 | 983.10 | 950.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 964.55 | 982.87 | 950.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 964.65 | 982.47 | 950.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 968.30 | 981.43 | 951.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:45:00 | 965.10 | 981.03 | 951.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 978.25 | 981.43 | 953.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 958.85 | 981.43 | 953.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 955.65 | 980.64 | 954.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:30:00 | 952.90 | 980.64 | 954.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 961.50 | 980.45 | 954.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 964.70 | 980.45 | 954.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 963.30 | 994.29 | 972.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 966.45 | 993.26 | 972.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 966.20 | 990.26 | 973.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 13:15:00 | 951.50 | 987.26 | 972.64 | SL hit (close<static) qty=1.00 sl=952.40 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 883.35 | 962.32 | 962.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 881.50 | 960.80 | 961.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 941.05 | 911.05 | 931.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 941.05 | 911.05 | 931.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 941.05 | 911.05 | 931.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:15:00 | 947.05 | 911.05 | 931.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 958.75 | 911.52 | 931.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 958.75 | 911.52 | 931.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1035.05 | 945.53 | 945.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 1038.50 | 946.46 | 945.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-22 09:15:00 | 755.00 | 2023-07-06 14:15:00 | 789.18 | PARTIAL | 0.50 | 4.53% |
| BUY | retest1 | 2023-06-27 09:30:00 | 751.60 | 2023-07-06 14:15:00 | 790.39 | PARTIAL | 0.50 | 5.16% |
| BUY | retest1 | 2023-06-22 09:15:00 | 755.00 | 2023-07-10 11:15:00 | 752.75 | STOP_HIT | 0.50 | -0.30% |
| BUY | retest1 | 2023-06-27 09:30:00 | 751.60 | 2023-07-10 11:15:00 | 752.75 | STOP_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2023-06-28 09:15:00 | 752.75 | 2023-07-24 09:15:00 | 733.95 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2023-06-28 14:15:00 | 756.00 | 2023-07-24 09:15:00 | 733.95 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-01-17 13:15:00 | 757.80 | 2024-01-18 12:15:00 | 737.80 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-01-17 14:00:00 | 757.20 | 2024-01-18 12:15:00 | 737.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-01-17 15:00:00 | 759.45 | 2024-01-18 12:15:00 | 737.80 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2024-04-05 11:15:00 | 626.95 | 2024-04-05 13:15:00 | 634.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-04-05 11:45:00 | 627.00 | 2024-04-05 13:15:00 | 634.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-04-08 13:15:00 | 626.40 | 2024-04-08 14:15:00 | 633.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-04-15 10:00:00 | 623.10 | 2024-04-15 14:15:00 | 633.45 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-04-16 11:15:00 | 613.45 | 2024-04-18 09:15:00 | 623.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-04-16 14:00:00 | 613.55 | 2024-04-18 09:15:00 | 623.95 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-04-19 09:15:00 | 612.80 | 2024-04-25 09:15:00 | 623.35 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-04-19 11:00:00 | 611.55 | 2024-04-25 09:15:00 | 623.35 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-04-22 10:15:00 | 606.25 | 2024-04-29 14:15:00 | 640.20 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2024-04-22 15:00:00 | 607.55 | 2024-04-29 14:15:00 | 640.20 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2024-04-24 10:00:00 | 606.50 | 2024-04-29 14:15:00 | 640.20 | STOP_HIT | 1.00 | -5.56% |
| SELL | retest2 | 2024-04-26 11:00:00 | 605.65 | 2024-04-29 14:15:00 | 640.20 | STOP_HIT | 1.00 | -5.70% |
| SELL | retest2 | 2024-05-22 10:15:00 | 611.65 | 2024-05-27 09:15:00 | 631.20 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-06-05 10:30:00 | 639.80 | 2024-07-25 12:15:00 | 642.65 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2024-07-12 09:15:00 | 638.10 | 2024-07-25 12:15:00 | 642.65 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-07-16 14:00:00 | 637.20 | 2024-07-25 12:15:00 | 642.65 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-07-22 10:00:00 | 638.40 | 2024-07-25 12:15:00 | 642.65 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-07-22 14:15:00 | 655.60 | 2024-07-26 11:15:00 | 643.55 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-23 10:15:00 | 652.10 | 2024-07-31 09:15:00 | 643.85 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-23 12:30:00 | 652.00 | 2024-07-31 09:15:00 | 643.85 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-07-23 13:15:00 | 652.50 | 2024-07-31 09:15:00 | 643.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-26 11:00:00 | 650.65 | 2024-08-08 14:15:00 | 625.35 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-07-26 14:00:00 | 650.10 | 2024-08-08 14:15:00 | 625.35 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2024-07-29 10:00:00 | 653.35 | 2024-08-08 14:15:00 | 625.35 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2024-07-29 12:45:00 | 650.00 | 2024-08-08 14:15:00 | 625.35 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-08-28 15:15:00 | 628.30 | 2024-08-29 11:15:00 | 649.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-10-14 13:30:00 | 692.90 | 2024-10-18 10:15:00 | 679.05 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-10-15 13:30:00 | 692.80 | 2024-10-18 10:15:00 | 679.05 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-16 12:00:00 | 692.55 | 2024-10-18 10:15:00 | 679.05 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-10-16 14:15:00 | 693.10 | 2024-10-18 10:15:00 | 679.05 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-01-23 11:30:00 | 591.50 | 2025-01-27 09:15:00 | 561.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 15:15:00 | 588.00 | 2025-01-27 09:15:00 | 558.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 591.50 | 2025-01-28 10:15:00 | 581.55 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2025-01-24 15:15:00 | 588.00 | 2025-01-28 10:15:00 | 581.55 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2025-01-28 14:45:00 | 591.10 | 2025-01-30 12:15:00 | 602.50 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-01-30 09:30:00 | 592.00 | 2025-01-30 12:15:00 | 602.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-02-06 11:15:00 | 585.95 | 2025-02-07 09:15:00 | 594.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-02-06 11:45:00 | 585.10 | 2025-02-07 09:15:00 | 594.95 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-02-06 14:45:00 | 586.15 | 2025-02-07 09:15:00 | 594.95 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-07 13:45:00 | 586.20 | 2025-02-12 09:15:00 | 556.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 585.30 | 2025-02-12 09:15:00 | 556.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 586.20 | 2025-02-17 09:15:00 | 527.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 585.30 | 2025-02-17 09:15:00 | 526.77 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-08 09:30:00 | 758.00 | 2025-08-08 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-18 09:45:00 | 760.85 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-19 09:45:00 | 757.70 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-20 12:15:00 | 758.05 | 2025-08-22 13:15:00 | 744.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-26 12:30:00 | 748.80 | 2025-08-26 13:15:00 | 742.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-26 13:15:00 | 749.50 | 2025-08-26 13:15:00 | 742.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 11:30:00 | 725.35 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-01 12:45:00 | 727.00 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-01 13:30:00 | 727.15 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-01 14:45:00 | 723.95 | 2025-10-03 09:15:00 | 735.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-01-28 09:15:00 | 964.55 | 2026-03-02 13:15:00 | 951.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-28 11:15:00 | 964.65 | 2026-03-02 13:15:00 | 951.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-29 09:15:00 | 968.30 | 2026-03-02 13:15:00 | 951.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-01-29 12:45:00 | 965.10 | 2026-03-02 13:15:00 | 951.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-02-02 14:15:00 | 964.70 | 2026-03-04 10:15:00 | 936.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-02-23 10:00:00 | 963.30 | 2026-03-04 10:15:00 | 936.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-23 13:15:00 | 966.45 | 2026-03-04 10:15:00 | 936.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2026-02-27 10:30:00 | 966.20 | 2026-03-04 10:15:00 | 936.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-03-06 09:15:00 | 977.90 | 2026-03-09 09:15:00 | 923.30 | STOP_HIT | 1.00 | -5.58% |
