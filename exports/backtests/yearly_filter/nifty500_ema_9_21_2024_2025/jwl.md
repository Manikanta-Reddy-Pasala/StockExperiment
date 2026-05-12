# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 298.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 122 |
| ALERT1 | 91 |
| ALERT2 | 89 |
| ALERT2_SKIP | 40 |
| ALERT3 | 224 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 107 |
| PARTIAL | 39 |
| TARGET_HIT | 22 |
| STOP_HIT | 87 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 92 / 56
- **Target hits / Stop hits / Partials:** 22 / 87 / 39
- **Avg / median % per leg:** 2.44% / 2.75%
- **Sum % (uncompounded):** 361.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 8 | 20.0% | 7 | 33 | 0 | 0.22% | 8.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.55% | -3.1% |
| BUY @ 3rd Alert (retest2) | 38 | 8 | 21.1% | 7 | 31 | 0 | 0.32% | 12.0% |
| SELL (all) | 108 | 84 | 77.8% | 15 | 54 | 39 | 3.27% | 352.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 108 | 84 | 77.8% | 15 | 54 | 39 | 3.27% | 352.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.55% | -3.1% |
| retest2 (combined) | 146 | 92 | 63.0% | 22 | 85 | 39 | 2.50% | 364.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 522.60 | 532.72 | 533.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 554.20 | 537.14 | 535.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 568.30 | 543.37 | 538.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 14:15:00 | 552.50 | 554.92 | 548.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 552.50 | 554.92 | 548.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 553.80 | 554.69 | 549.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 553.05 | 554.69 | 549.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 550.95 | 553.94 | 549.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:30:00 | 558.00 | 552.28 | 549.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:30:00 | 557.95 | 554.82 | 551.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 10:00:00 | 558.00 | 567.41 | 562.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 558.70 | 566.37 | 562.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 584.50 | 569.99 | 564.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 14:00:00 | 591.50 | 576.00 | 568.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 599.40 | 580.68 | 571.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-31 15:15:00 | 613.80 | 585.34 | 574.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 571.50 | 597.27 | 599.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 499.30 | 554.29 | 576.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 539.55 | 538.31 | 558.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 539.55 | 538.31 | 558.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 605.10 | 552.74 | 561.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 605.10 | 552.74 | 561.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 572.65 | 564.79 | 565.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 575.00 | 564.79 | 565.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 580.95 | 568.02 | 566.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 585.60 | 573.05 | 569.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 12:15:00 | 694.40 | 696.13 | 682.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 13:00:00 | 694.40 | 696.13 | 682.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 691.70 | 693.92 | 683.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:45:00 | 686.50 | 693.92 | 683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 685.60 | 690.98 | 684.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:30:00 | 702.00 | 691.68 | 685.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:00:00 | 694.50 | 691.68 | 685.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 674.00 | 685.93 | 684.92 | SL hit (close<static) qty=1.00 sl=675.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 676.80 | 682.77 | 683.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 664.00 | 675.54 | 679.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 694.25 | 674.90 | 677.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 13:15:00 | 694.25 | 674.90 | 677.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 694.25 | 674.90 | 677.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 694.25 | 674.90 | 677.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 691.95 | 678.31 | 678.76 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 15:15:00 | 691.00 | 680.85 | 679.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 701.90 | 685.06 | 681.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 690.00 | 693.21 | 688.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 690.00 | 693.21 | 688.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 690.00 | 693.21 | 688.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 687.60 | 693.21 | 688.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 688.90 | 692.35 | 688.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 687.40 | 692.35 | 688.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 687.55 | 691.39 | 688.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 687.85 | 691.39 | 688.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 687.40 | 690.59 | 688.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:45:00 | 686.05 | 690.59 | 688.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 683.00 | 687.98 | 687.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:45:00 | 686.00 | 687.98 | 687.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 680.35 | 686.45 | 687.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 674.60 | 684.08 | 685.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 673.65 | 669.90 | 676.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 673.65 | 669.90 | 676.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 673.65 | 669.90 | 676.48 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 687.90 | 678.27 | 678.06 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 671.00 | 677.95 | 678.16 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 681.00 | 678.56 | 678.42 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 676.45 | 678.37 | 678.37 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 682.00 | 679.09 | 678.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 689.90 | 682.45 | 680.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 689.05 | 689.15 | 685.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:15:00 | 694.65 | 689.15 | 685.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 685.45 | 688.54 | 685.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 685.45 | 688.54 | 685.95 | SL hit (close<ema400) qty=1.00 sl=685.95 alert=retest1 |

### Cycle 13 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 680.70 | 684.61 | 684.69 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 695.00 | 685.88 | 685.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 700.05 | 688.71 | 686.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 714.95 | 724.34 | 715.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 714.95 | 724.34 | 715.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 714.95 | 724.34 | 715.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 714.95 | 724.34 | 715.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 708.50 | 721.17 | 715.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 708.50 | 721.17 | 715.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 718.00 | 720.54 | 715.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:30:00 | 719.00 | 721.37 | 716.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 701.45 | 718.05 | 716.01 | SL hit (close<static) qty=1.00 sl=707.70 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 701.05 | 714.65 | 714.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 678.35 | 701.38 | 707.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 11:15:00 | 698.05 | 690.22 | 696.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 698.05 | 690.22 | 696.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 698.05 | 690.22 | 696.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 698.05 | 690.22 | 696.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 694.85 | 691.14 | 695.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 13:15:00 | 691.95 | 691.14 | 695.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:00:00 | 691.00 | 691.11 | 695.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:45:00 | 690.90 | 690.89 | 695.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 700.95 | 692.60 | 695.05 | SL hit (close>static) qty=1.00 sl=699.80 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 700.00 | 696.78 | 696.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 714.50 | 700.36 | 698.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 702.20 | 702.98 | 700.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 702.20 | 702.98 | 700.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 705.00 | 703.38 | 701.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 667.10 | 703.38 | 701.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 664.35 | 695.57 | 697.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 650.60 | 666.86 | 679.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 639.00 | 626.68 | 640.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 639.00 | 626.68 | 640.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 639.00 | 626.68 | 640.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 638.00 | 626.68 | 640.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 640.25 | 629.40 | 640.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 643.80 | 629.40 | 640.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 639.85 | 631.49 | 640.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 636.00 | 633.90 | 639.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 15:15:00 | 642.50 | 636.89 | 640.23 | SL hit (close>static) qty=1.00 sl=641.90 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 644.00 | 637.46 | 636.66 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 628.70 | 636.76 | 636.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 625.45 | 633.24 | 635.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 11:15:00 | 627.85 | 627.52 | 631.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 12:00:00 | 627.85 | 627.52 | 631.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 623.15 | 622.39 | 627.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 625.40 | 622.39 | 627.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 618.00 | 621.52 | 626.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:00:00 | 615.60 | 620.33 | 625.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 616.00 | 619.10 | 622.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:30:00 | 616.70 | 618.08 | 621.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 584.82 | 602.42 | 608.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 585.20 | 602.42 | 608.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 585.87 | 602.42 | 608.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 609.50 | 603.83 | 608.30 | SL hit (close>ema200) qty=0.50 sl=603.83 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 574.00 | 562.13 | 561.69 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 547.25 | 561.08 | 562.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 09:15:00 | 536.00 | 548.13 | 552.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 11:15:00 | 543.00 | 538.59 | 543.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 543.00 | 538.59 | 543.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 543.00 | 538.59 | 543.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 543.00 | 538.59 | 543.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 542.70 | 539.41 | 543.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:15:00 | 539.00 | 540.29 | 543.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 540.00 | 540.23 | 542.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 541.30 | 541.03 | 542.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 566.95 | 546.69 | 544.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 566.95 | 546.69 | 544.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 15:15:00 | 568.00 | 554.22 | 548.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 553.60 | 561.74 | 556.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 14:15:00 | 553.60 | 561.74 | 556.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 553.60 | 561.74 | 556.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 553.60 | 561.74 | 556.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 559.35 | 561.26 | 556.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 556.00 | 561.26 | 556.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 553.00 | 559.61 | 556.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 553.90 | 559.61 | 556.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 556.10 | 558.91 | 556.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 553.15 | 558.91 | 556.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 554.80 | 558.08 | 556.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 554.80 | 558.08 | 556.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 557.20 | 557.91 | 556.16 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 547.40 | 554.67 | 555.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 542.20 | 552.18 | 553.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 553.60 | 549.39 | 551.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 553.60 | 549.39 | 551.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 553.60 | 549.39 | 551.45 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 556.95 | 553.13 | 552.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 13:15:00 | 558.80 | 554.27 | 553.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 15:15:00 | 559.00 | 560.13 | 557.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:15:00 | 566.00 | 560.13 | 557.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 563.60 | 562.22 | 559.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 568.15 | 562.22 | 559.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 556.00 | 561.83 | 560.09 | SL hit (close<ema400) qty=1.00 sl=560.09 alert=retest1 |

### Cycle 25 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 550.30 | 562.08 | 563.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 09:15:00 | 547.40 | 556.52 | 560.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 556.65 | 551.60 | 555.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 556.65 | 551.60 | 555.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 556.65 | 551.60 | 555.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 556.65 | 551.60 | 555.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 553.60 | 552.00 | 554.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 13:00:00 | 551.70 | 552.39 | 554.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 551.05 | 552.36 | 554.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:00:00 | 551.95 | 553.60 | 554.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:00:00 | 550.15 | 553.76 | 554.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 549.85 | 552.71 | 553.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 522.15 | 550.55 | 552.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 524.12 | 545.04 | 549.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 523.50 | 545.04 | 549.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 524.35 | 545.04 | 549.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 522.64 | 545.04 | 549.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-10 09:15:00 | 496.53 | 515.00 | 529.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 530.70 | 522.90 | 522.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 539.70 | 527.80 | 525.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 546.25 | 549.24 | 539.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 546.25 | 549.24 | 539.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 539.35 | 545.42 | 542.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 539.35 | 545.42 | 542.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 529.75 | 542.28 | 541.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 529.75 | 542.28 | 541.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 525.00 | 538.83 | 539.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 515.30 | 527.94 | 531.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 541.40 | 525.00 | 527.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 541.40 | 525.00 | 527.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 541.40 | 525.00 | 527.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 537.35 | 525.00 | 527.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 539.70 | 527.94 | 528.73 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 538.35 | 530.02 | 529.60 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 13:15:00 | 525.95 | 529.20 | 529.30 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 539.40 | 531.20 | 530.19 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 528.55 | 532.38 | 532.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 523.00 | 529.63 | 531.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 13:15:00 | 534.75 | 526.93 | 528.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 13:15:00 | 534.75 | 526.93 | 528.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 534.75 | 526.93 | 528.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 534.75 | 526.93 | 528.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 532.45 | 528.03 | 528.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 533.30 | 528.03 | 528.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 531.20 | 528.67 | 529.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 525.25 | 528.67 | 529.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 525.50 | 527.38 | 528.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 526.70 | 527.38 | 528.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 518.55 | 514.67 | 517.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 513.90 | 514.67 | 517.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:45:00 | 514.65 | 514.51 | 516.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 488.20 | 503.07 | 507.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 504.55 | 503.07 | 507.92 | SL hit (close>static) qty=0.50 sl=503.07 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 500.05 | 487.62 | 487.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 504.80 | 491.06 | 488.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 500.45 | 500.67 | 496.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:30:00 | 501.55 | 500.67 | 496.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 515.70 | 503.82 | 499.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 526.15 | 510.16 | 505.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 522.70 | 511.31 | 506.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 518.95 | 517.81 | 513.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:15:00 | 519.40 | 517.46 | 513.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 517.15 | 517.73 | 515.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 516.65 | 517.73 | 515.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 513.95 | 516.97 | 515.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 513.95 | 516.97 | 515.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 511.60 | 515.90 | 514.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:00:00 | 511.60 | 515.90 | 514.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 514.35 | 515.59 | 514.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:15:00 | 517.00 | 515.59 | 514.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:15:00 | 516.50 | 515.86 | 514.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:00:00 | 517.10 | 521.61 | 519.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 12:15:00 | 516.90 | 519.72 | 519.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 12:15:00 | 516.55 | 519.09 | 519.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 516.55 | 519.09 | 519.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 510.90 | 516.70 | 518.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 485.40 | 483.40 | 494.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 485.40 | 483.40 | 494.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 469.90 | 465.38 | 469.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 469.90 | 465.38 | 469.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 466.00 | 465.50 | 468.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 475.30 | 465.50 | 468.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 468.20 | 466.04 | 468.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 462.80 | 465.18 | 468.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 461.75 | 464.75 | 467.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 481.55 | 468.22 | 468.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 481.55 | 468.22 | 468.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 485.40 | 473.84 | 470.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 504.70 | 511.73 | 500.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 504.70 | 511.73 | 500.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 500.00 | 508.40 | 502.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:45:00 | 502.55 | 508.40 | 502.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 500.05 | 506.73 | 502.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 501.95 | 506.73 | 502.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 495.50 | 503.58 | 501.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 495.50 | 503.58 | 501.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 493.00 | 501.46 | 501.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 491.60 | 501.46 | 501.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 492.90 | 499.75 | 500.30 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 507.65 | 500.58 | 500.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 512.00 | 504.48 | 502.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 513.35 | 513.71 | 509.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:00:00 | 513.35 | 513.71 | 509.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 509.00 | 512.30 | 509.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 503.20 | 512.30 | 509.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 503.00 | 510.44 | 509.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 501.70 | 510.44 | 509.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 504.05 | 509.16 | 508.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 501.75 | 509.16 | 508.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 498.35 | 507.00 | 507.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 493.50 | 501.63 | 504.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 15:15:00 | 437.50 | 437.45 | 444.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:15:00 | 447.35 | 437.45 | 444.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 453.00 | 440.56 | 444.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 453.00 | 440.56 | 444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 450.70 | 442.59 | 445.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 446.25 | 445.16 | 446.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 423.94 | 431.08 | 436.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 446.60 | 431.56 | 433.61 | SL hit (close>ema200) qty=0.50 sl=431.56 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 444.90 | 436.43 | 435.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 12:15:00 | 448.05 | 438.75 | 436.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 484.20 | 486.26 | 476.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:15:00 | 486.90 | 486.26 | 476.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 492.00 | 495.89 | 491.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:00:00 | 492.00 | 495.89 | 491.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 490.05 | 494.72 | 491.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:30:00 | 489.75 | 494.72 | 491.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 490.50 | 493.88 | 491.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 508.00 | 493.88 | 491.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:15:00 | 494.50 | 494.46 | 492.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:30:00 | 492.20 | 494.17 | 492.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 487.95 | 492.34 | 492.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 487.95 | 492.34 | 492.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 11:15:00 | 486.55 | 491.18 | 491.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 493.45 | 488.53 | 489.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 10:15:00 | 493.45 | 488.53 | 489.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 493.45 | 488.53 | 489.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 493.45 | 488.53 | 489.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 491.20 | 489.07 | 489.86 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 494.20 | 490.66 | 490.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 499.00 | 492.97 | 491.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 496.25 | 499.37 | 496.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 496.25 | 499.37 | 496.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 496.25 | 499.37 | 496.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 496.25 | 499.37 | 496.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 495.30 | 498.56 | 496.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:45:00 | 496.65 | 496.54 | 495.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-11 10:15:00 | 546.32 | 513.17 | 503.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 547.90 | 550.57 | 550.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 537.25 | 547.91 | 549.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 535.75 | 530.26 | 535.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 14:15:00 | 535.75 | 530.26 | 535.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 535.75 | 530.26 | 535.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 535.75 | 530.26 | 535.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 539.00 | 532.01 | 536.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 531.95 | 532.01 | 536.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:15:00 | 534.10 | 530.52 | 533.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:15:00 | 505.35 | 512.06 | 517.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:15:00 | 507.39 | 512.06 | 517.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 506.00 | 497.63 | 501.93 | SL hit (close>ema200) qty=0.50 sl=497.63 alert=retest2 |

### Cycle 42 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 506.25 | 503.55 | 503.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 521.00 | 507.84 | 505.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 510.75 | 511.71 | 508.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 510.75 | 511.71 | 508.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 509.20 | 510.96 | 508.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 500.60 | 510.96 | 508.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 499.00 | 508.57 | 507.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 499.00 | 508.57 | 507.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 484.85 | 503.83 | 505.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 479.40 | 493.20 | 499.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 486.55 | 486.37 | 493.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 11:15:00 | 486.55 | 486.37 | 493.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 486.55 | 486.37 | 493.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 490.65 | 486.37 | 493.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 489.20 | 488.13 | 492.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 481.25 | 488.20 | 492.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 457.19 | 469.92 | 477.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 433.12 | 445.56 | 458.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 472.10 | 445.97 | 443.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 487.45 | 454.27 | 447.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 492.20 | 496.83 | 487.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 492.20 | 496.83 | 487.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 489.25 | 493.16 | 488.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 489.15 | 493.16 | 488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 488.65 | 492.26 | 488.12 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 470.30 | 485.17 | 485.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 466.30 | 481.40 | 483.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 473.75 | 472.87 | 478.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 473.75 | 472.87 | 478.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 481.70 | 474.50 | 478.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 481.70 | 474.50 | 478.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 482.20 | 476.04 | 478.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 483.30 | 476.04 | 478.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 476.35 | 477.48 | 478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 474.20 | 477.48 | 478.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 473.80 | 476.74 | 478.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 450.49 | 464.34 | 471.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 450.11 | 464.34 | 471.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-24 13:15:00 | 426.78 | 457.67 | 467.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 403.90 | 387.13 | 386.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 426.50 | 395.01 | 389.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 382.40 | 398.55 | 393.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 382.40 | 398.55 | 393.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 382.40 | 398.55 | 393.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 382.40 | 398.55 | 393.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 385.60 | 395.96 | 392.66 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 378.55 | 389.51 | 390.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 354.00 | 382.41 | 386.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 360.80 | 355.71 | 362.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 360.80 | 355.71 | 362.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 360.80 | 355.71 | 362.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 366.80 | 355.71 | 362.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 355.55 | 357.13 | 360.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 352.85 | 355.86 | 359.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 335.21 | 344.16 | 349.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 317.57 | 332.55 | 340.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 325.45 | 310.33 | 309.15 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 308.85 | 314.96 | 315.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 299.00 | 305.97 | 309.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 284.60 | 283.71 | 290.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 284.60 | 283.71 | 290.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 296.35 | 286.36 | 290.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 296.75 | 286.36 | 290.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 297.85 | 288.66 | 290.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 297.85 | 288.66 | 290.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 292.20 | 290.00 | 290.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 292.20 | 290.00 | 290.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 292.55 | 290.51 | 291.13 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 303.65 | 293.35 | 292.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 307.85 | 303.55 | 299.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 307.80 | 312.62 | 308.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 307.80 | 312.62 | 308.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 307.80 | 312.62 | 308.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 307.80 | 312.62 | 308.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 307.15 | 311.52 | 308.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:15:00 | 307.80 | 311.52 | 308.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 306.50 | 310.52 | 308.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 306.50 | 310.52 | 308.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 305.65 | 309.54 | 308.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 305.55 | 309.54 | 308.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 298.95 | 305.65 | 306.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 293.55 | 303.23 | 305.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 300.75 | 298.91 | 301.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 300.90 | 298.91 | 301.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 297.70 | 298.67 | 301.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 294.50 | 298.21 | 300.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 295.00 | 296.36 | 298.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 295.95 | 295.97 | 298.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 294.75 | 295.88 | 297.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 294.70 | 295.15 | 296.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 295.30 | 295.15 | 296.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 294.85 | 295.09 | 296.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 294.85 | 295.09 | 296.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 298.00 | 293.71 | 295.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 298.95 | 295.95 | 295.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 298.95 | 295.95 | 295.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 302.60 | 297.28 | 296.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 12:15:00 | 368.30 | 370.16 | 364.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 368.30 | 370.16 | 364.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 365.80 | 369.48 | 367.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 365.80 | 369.48 | 367.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 369.75 | 369.54 | 367.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 372.80 | 369.50 | 368.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 372.10 | 369.97 | 368.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 364.50 | 369.22 | 368.68 | SL hit (close<static) qty=1.00 sl=365.05 alert=retest2 |

### Cycle 53 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 368.15 | 370.27 | 370.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 364.90 | 368.70 | 369.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 369.00 | 368.76 | 369.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 14:15:00 | 369.00 | 368.76 | 369.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 369.00 | 368.76 | 369.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 369.00 | 368.76 | 369.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 368.00 | 368.61 | 369.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 342.95 | 368.61 | 369.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 325.80 | 362.69 | 366.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 358.55 | 356.27 | 361.75 | SL hit (close>ema200) qty=0.50 sl=356.27 alert=retest2 |

### Cycle 54 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 371.60 | 365.37 | 364.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 373.85 | 369.28 | 367.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 367.80 | 370.26 | 368.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 367.80 | 370.26 | 368.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 367.80 | 370.26 | 368.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:30:00 | 367.80 | 370.26 | 368.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 365.70 | 369.35 | 367.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 365.70 | 369.35 | 367.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 365.85 | 368.65 | 367.73 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 15:15:00 | 359.00 | 365.86 | 366.65 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 371.80 | 367.77 | 367.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 376.50 | 370.87 | 369.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 381.00 | 382.04 | 378.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 389.50 | 384.82 | 381.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 389.50 | 384.82 | 381.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 390.65 | 384.82 | 381.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:00:00 | 390.55 | 385.96 | 382.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 394.80 | 388.52 | 385.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:00:00 | 391.00 | 391.70 | 388.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 387.25 | 390.89 | 388.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 387.25 | 390.89 | 388.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 385.30 | 389.77 | 388.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 376.60 | 387.14 | 387.11 | SL hit (close<static) qty=1.00 sl=381.65 alert=retest2 |

### Cycle 57 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 378.40 | 385.39 | 386.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 371.15 | 381.71 | 383.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 371.00 | 368.54 | 371.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 371.00 | 368.54 | 371.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 371.00 | 368.54 | 371.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 371.00 | 368.54 | 371.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 370.00 | 368.83 | 371.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 361.45 | 368.83 | 371.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 343.38 | 350.01 | 354.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 343.15 | 342.40 | 346.79 | SL hit (close>ema200) qty=0.50 sl=342.40 alert=retest2 |

### Cycle 58 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 360.95 | 344.64 | 343.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 367.00 | 354.07 | 348.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 410.35 | 429.45 | 415.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 408.95 | 425.35 | 415.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 395.50 | 425.35 | 415.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 398.85 | 415.36 | 412.10 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 395.85 | 408.90 | 409.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 394.65 | 406.05 | 408.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 403.80 | 401.96 | 405.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 403.80 | 401.96 | 405.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 400.25 | 398.93 | 401.70 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 402.90 | 399.61 | 399.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 411.10 | 402.97 | 401.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 406.20 | 407.22 | 404.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:15:00 | 401.80 | 407.22 | 404.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 403.45 | 406.47 | 404.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 403.30 | 406.47 | 404.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 403.70 | 405.91 | 404.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 402.95 | 405.91 | 404.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 404.40 | 405.91 | 405.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 404.40 | 405.91 | 405.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 405.30 | 405.79 | 405.04 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 398.65 | 404.44 | 404.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 397.10 | 400.61 | 402.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 399.80 | 399.05 | 401.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 402.50 | 399.69 | 400.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 396.25 | 399.25 | 400.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 11:15:00 | 412.80 | 398.10 | 397.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 412.80 | 398.10 | 397.84 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 399.95 | 402.29 | 402.44 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 408.10 | 403.43 | 402.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 409.90 | 406.62 | 404.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 408.00 | 408.35 | 406.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 407.25 | 408.35 | 406.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 410.65 | 408.81 | 407.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 407.50 | 408.81 | 407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 405.50 | 408.78 | 407.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 404.55 | 408.78 | 407.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 409.10 | 408.84 | 407.76 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 403.10 | 406.60 | 406.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 393.20 | 403.43 | 405.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 389.30 | 389.23 | 393.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 389.30 | 389.23 | 393.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 389.75 | 389.95 | 392.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 392.00 | 389.95 | 392.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 386.60 | 387.04 | 389.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 384.25 | 386.93 | 389.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 384.45 | 386.93 | 389.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 365.04 | 376.07 | 381.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 365.23 | 376.07 | 381.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 373.35 | 371.34 | 376.74 | SL hit (close>ema200) qty=0.50 sl=371.34 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 383.30 | 375.88 | 375.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 388.40 | 384.99 | 382.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 387.45 | 387.72 | 385.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 387.45 | 387.72 | 385.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 385.05 | 387.23 | 386.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 385.85 | 387.23 | 386.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 384.20 | 386.62 | 386.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 384.20 | 386.62 | 386.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 384.55 | 385.64 | 385.73 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 387.50 | 386.10 | 385.93 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 384.00 | 385.68 | 385.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 383.35 | 384.65 | 385.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 383.25 | 383.24 | 383.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:45:00 | 383.15 | 383.24 | 383.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 381.40 | 381.66 | 382.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:30:00 | 379.70 | 380.78 | 381.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 378.85 | 376.42 | 377.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 378.75 | 376.92 | 377.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 379.00 | 376.92 | 377.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 376.70 | 376.88 | 377.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 375.75 | 376.65 | 377.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 372.50 | 371.41 | 371.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 372.50 | 371.41 | 371.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 373.90 | 371.91 | 371.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 372.15 | 372.09 | 371.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 371.75 | 372.02 | 371.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 371.75 | 372.02 | 371.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 370.40 | 371.70 | 371.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 370.40 | 371.70 | 371.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 370.30 | 371.42 | 371.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 369.95 | 370.98 | 371.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 367.95 | 367.75 | 369.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 367.50 | 367.70 | 368.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 371.25 | 368.41 | 369.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 371.25 | 368.41 | 369.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 369.75 | 368.68 | 369.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 369.30 | 368.92 | 369.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 369.35 | 369.17 | 369.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 369.00 | 369.17 | 369.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.83 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.88 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.55 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 346.15 | 345.01 | 349.32 | SL hit (close>ema200) qty=0.50 sl=345.01 alert=retest2 |

### Cycle 72 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 345.15 | 342.52 | 342.24 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 337.55 | 342.01 | 342.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 333.25 | 337.73 | 339.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 338.35 | 335.08 | 337.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 339.00 | 335.87 | 337.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 337.25 | 335.72 | 337.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 320.39 | 328.41 | 331.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 325.10 | 317.45 | 320.35 | SL hit (close>ema200) qty=0.50 sl=317.45 alert=retest2 |

### Cycle 74 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 324.80 | 321.76 | 321.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 326.25 | 322.66 | 322.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 327.90 | 328.53 | 326.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 327.90 | 328.53 | 326.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 341.40 | 348.11 | 340.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 338.30 | 348.11 | 340.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 340.65 | 346.62 | 340.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 341.05 | 346.62 | 340.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 340.10 | 345.32 | 340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 340.10 | 345.32 | 340.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 339.65 | 344.18 | 340.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 339.65 | 344.18 | 340.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 340.00 | 343.35 | 340.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 339.25 | 343.35 | 340.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 340.65 | 342.24 | 340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 339.50 | 342.24 | 340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 340.65 | 341.92 | 340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 335.50 | 341.92 | 340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 336.20 | 340.78 | 340.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 336.30 | 340.78 | 340.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 336.75 | 339.97 | 339.87 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 336.75 | 339.33 | 339.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 336.40 | 338.74 | 339.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 322.75 | 322.74 | 327.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 322.40 | 322.74 | 327.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 323.40 | 321.01 | 322.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 324.75 | 321.01 | 322.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 327.75 | 322.36 | 323.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 328.40 | 322.36 | 323.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 330.40 | 323.97 | 323.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 335.25 | 327.35 | 325.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 330.20 | 330.23 | 328.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 328.60 | 330.23 | 328.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 329.10 | 330.01 | 328.64 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 326.10 | 327.95 | 327.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 325.50 | 327.46 | 327.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 324.80 | 324.36 | 325.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 326.80 | 324.36 | 325.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 332.20 | 325.93 | 326.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 331.50 | 325.93 | 326.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 329.20 | 326.58 | 326.47 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 325.00 | 326.36 | 326.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 323.45 | 325.63 | 326.13 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 334.60 | 323.20 | 323.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 346.00 | 335.25 | 331.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 341.65 | 342.31 | 337.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:15:00 | 340.85 | 342.31 | 337.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 342.60 | 344.11 | 343.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 342.60 | 344.11 | 343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 342.80 | 343.85 | 343.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 343.70 | 343.85 | 343.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 343.80 | 343.84 | 343.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:00:00 | 343.75 | 343.82 | 343.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 339.80 | 342.93 | 342.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 339.80 | 342.93 | 342.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 337.75 | 340.92 | 341.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 334.45 | 327.41 | 329.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 341.40 | 330.21 | 330.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 344.30 | 330.21 | 330.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 11:15:00 | 334.80 | 331.13 | 330.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 341.70 | 339.43 | 338.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 340.80 | 341.13 | 339.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 342.40 | 341.13 | 339.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 338.15 | 340.54 | 339.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 338.15 | 340.54 | 339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 339.60 | 340.35 | 339.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:30:00 | 339.90 | 340.37 | 339.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 336.00 | 339.24 | 339.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 336.00 | 339.24 | 339.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 335.00 | 337.97 | 338.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 335.85 | 335.72 | 337.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 335.85 | 335.72 | 337.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 336.05 | 335.90 | 336.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 331.50 | 335.11 | 336.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 323.70 | 322.82 | 322.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 323.70 | 322.82 | 322.73 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 321.35 | 322.61 | 322.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 320.85 | 321.92 | 322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 323.60 | 321.99 | 322.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 322.80 | 322.15 | 322.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 324.80 | 322.15 | 322.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 323.30 | 322.38 | 322.37 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 321.50 | 322.22 | 322.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 320.85 | 321.95 | 322.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 323.75 | 321.32 | 321.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 323.85 | 321.83 | 321.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 323.85 | 321.83 | 321.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 325.65 | 322.59 | 322.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 327.25 | 324.03 | 323.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 326.50 | 326.88 | 325.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 326.50 | 326.88 | 325.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 326.50 | 326.81 | 325.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 325.85 | 326.81 | 325.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 324.80 | 326.21 | 325.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 324.80 | 326.21 | 325.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 322.80 | 325.53 | 325.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 322.80 | 325.53 | 325.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 323.45 | 324.88 | 324.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 322.70 | 324.23 | 324.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 322.30 | 323.37 | 323.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 14:15:00 | 306.19 | 313.10 | 313.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 313.75 | 313.10 | 313.98 | SL hit (close>static) qty=0.50 sl=313.10 alert=retest2 |

### Cycle 90 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 310.40 | 308.16 | 308.15 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 305.05 | 308.11 | 308.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 303.55 | 305.55 | 306.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 304.65 | 303.32 | 304.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 291.05 | 290.27 | 292.69 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 293.40 | 292.81 | 292.80 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 292.00 | 292.65 | 292.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 289.10 | 291.80 | 292.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 289.25 | 288.82 | 290.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 287.10 | 288.47 | 289.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:15:00 | 286.50 | 288.47 | 289.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 272.18 | 275.54 | 278.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 10:15:00 | 257.85 | 266.43 | 271.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 94 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 261.05 | 258.40 | 258.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 261.85 | 260.11 | 259.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 263.65 | 265.32 | 262.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 264.85 | 265.32 | 262.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 263.55 | 264.60 | 262.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 262.50 | 264.60 | 262.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 263.35 | 264.35 | 263.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 263.70 | 264.35 | 263.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 263.15 | 264.11 | 263.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:15:00 | 262.60 | 264.11 | 263.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 262.80 | 263.85 | 263.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 262.75 | 263.85 | 263.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 262.75 | 263.63 | 262.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 261.80 | 263.63 | 262.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 259.70 | 262.84 | 262.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 259.70 | 262.84 | 262.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 259.35 | 262.14 | 262.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 258.35 | 260.92 | 261.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 256.85 | 255.71 | 257.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 255.65 | 255.70 | 257.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 257.30 | 255.70 | 257.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 259.60 | 256.14 | 257.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 258.85 | 256.14 | 257.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 260.15 | 256.94 | 257.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 286.20 | 256.94 | 257.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 290.90 | 263.73 | 260.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 292.05 | 269.40 | 263.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 345.20 | 345.64 | 335.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 342.55 | 345.64 | 335.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 338.80 | 341.87 | 335.98 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 332.60 | 334.64 | 334.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 331.10 | 333.93 | 334.48 | Break + close below crossover candle low |

### Cycle 98 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 347.30 | 336.10 | 335.34 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 336.85 | 337.38 | 337.41 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 338.15 | 337.38 | 337.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 339.95 | 338.20 | 337.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 337.85 | 338.13 | 337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 339.00 | 338.30 | 337.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 337.50 | 338.30 | 337.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 333.35 | 337.31 | 337.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 330.55 | 335.96 | 336.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 295.20 | 294.42 | 302.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 294.55 | 294.42 | 302.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 296.65 | 295.85 | 300.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 292.40 | 294.77 | 298.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 326.20 | 301.43 | 300.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 326.20 | 301.43 | 300.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 330.90 | 311.15 | 304.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 319.90 | 321.52 | 315.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 323.00 | 321.52 | 315.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 326.40 | 322.50 | 316.87 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 315.50 | 316.86 | 317.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 309.95 | 315.48 | 316.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:30:00 | 315.00 | 313.88 | 315.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 311.30 | 313.36 | 314.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 305.00 | 309.38 | 311.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:45:00 | 303.55 | 307.79 | 309.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:45:00 | 305.15 | 307.19 | 309.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 314.15 | 309.82 | 309.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 314.15 | 309.82 | 309.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 321.10 | 312.08 | 310.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 324.65 | 324.72 | 320.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 324.65 | 324.72 | 320.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 321.50 | 325.96 | 323.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 321.50 | 325.96 | 323.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 315.35 | 323.84 | 322.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 314.55 | 323.84 | 322.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 310.30 | 321.13 | 321.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 301.50 | 317.21 | 319.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 308.25 | 307.02 | 312.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 308.25 | 307.02 | 312.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 309.30 | 307.64 | 311.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 307.30 | 309.53 | 310.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 309.05 | 304.04 | 303.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 309.05 | 304.04 | 303.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 313.70 | 306.71 | 304.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 310.00 | 310.71 | 308.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 307.90 | 310.71 | 308.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 307.40 | 310.05 | 308.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 306.45 | 310.05 | 308.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 307.50 | 309.54 | 307.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 307.50 | 309.54 | 307.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 307.65 | 309.16 | 307.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 307.65 | 309.16 | 307.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 308.50 | 309.03 | 307.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 307.35 | 309.03 | 307.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 308.50 | 308.92 | 308.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 308.50 | 308.92 | 308.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 310.95 | 309.33 | 308.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:15:00 | 308.00 | 309.33 | 308.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 308.00 | 309.06 | 308.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 297.70 | 309.06 | 308.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 299.90 | 307.23 | 307.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 300.00 | 303.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 292.80 | 291.26 | 294.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 293.85 | 291.26 | 294.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 293.75 | 291.76 | 294.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 289.05 | 290.81 | 292.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 288.50 | 291.44 | 291.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 288.90 | 289.82 | 290.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 289.25 | 289.28 | 290.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 287.95 | 289.07 | 289.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 287.00 | 289.07 | 289.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 287.25 | 288.16 | 289.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 287.30 | 288.01 | 289.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 285.70 | 288.64 | 289.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 285.40 | 287.99 | 288.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:30:00 | 284.30 | 286.73 | 288.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 283.40 | 285.44 | 287.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.60 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.07 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.45 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.79 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 280.95 | 280.34 | 282.92 | SL hit (close>ema200) qty=0.50 sl=280.34 alert=retest2 |

### Cycle 108 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 286.60 | 259.30 | 258.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 287.95 | 265.03 | 260.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 13:15:00 | 285.70 | 288.39 | 280.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 285.70 | 288.39 | 280.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 280.55 | 286.22 | 281.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 277.75 | 286.22 | 281.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 283.75 | 285.73 | 281.42 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 278.90 | 281.79 | 281.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 277.15 | 280.86 | 281.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 262.55 | 261.68 | 266.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 264.85 | 261.68 | 266.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 263.25 | 261.88 | 265.37 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 278.50 | 266.21 | 265.80 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 263.25 | 267.08 | 267.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 262.30 | 266.12 | 267.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 263.30 | 265.28 | 266.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 263.95 | 264.50 | 265.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 250.75 | 259.80 | 263.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 250.13 | 256.37 | 260.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 256.20 | 255.08 | 258.02 | SL hit (close>ema200) qty=0.50 sl=255.08 alert=retest2 |

### Cycle 112 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 265.90 | 259.47 | 259.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 268.45 | 261.27 | 260.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 261.70 | 262.70 | 261.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 261.70 | 262.70 | 261.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 261.60 | 262.48 | 261.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 259.00 | 262.48 | 261.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 254.30 | 260.85 | 260.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 254.30 | 260.85 | 260.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 252.25 | 259.13 | 259.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 250.80 | 255.26 | 257.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 255.90 | 245.95 | 250.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 251.93 | 247.15 | 250.27 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 255.75 | 252.33 | 252.06 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 248.10 | 251.42 | 251.76 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 256.98 | 252.83 | 252.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 258.03 | 255.42 | 253.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 266.25 | 266.62 | 261.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 266.25 | 266.62 | 261.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 269.15 | 270.87 | 268.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 269.22 | 270.87 | 268.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 269.52 | 270.60 | 268.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 269.52 | 270.60 | 268.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 269.50 | 270.38 | 268.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 272.94 | 270.38 | 268.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 268.00 | 269.63 | 269.24 | SL hit (close<static) qty=1.00 sl=268.08 alert=retest2 |

### Cycle 117 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 260.00 | 267.70 | 268.40 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 273.85 | 267.30 | 267.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 282.80 | 273.86 | 270.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 279.30 | 284.20 | 280.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 280.55 | 282.81 | 280.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 280.61 | 282.81 | 280.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 279.86 | 282.22 | 280.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 279.86 | 282.22 | 280.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 279.00 | 281.58 | 280.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 281.44 | 281.58 | 280.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:00:00 | 282.00 | 285.13 | 284.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 278.55 | 283.81 | 283.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 278.55 | 283.81 | 283.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 277.65 | 281.85 | 282.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 283.99 | 280.99 | 282.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 286.80 | 282.15 | 282.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 287.02 | 282.15 | 282.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 285.76 | 282.87 | 282.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 288.05 | 283.91 | 283.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 289.10 | 289.22 | 287.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 289.10 | 289.22 | 287.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 287.62 | 289.35 | 288.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 287.62 | 289.35 | 288.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 287.40 | 288.96 | 288.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 287.40 | 288.96 | 288.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 287.50 | 288.67 | 288.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 283.85 | 288.67 | 288.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 281.39 | 287.21 | 287.56 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 287.75 | 286.35 | 286.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 289.70 | 287.02 | 286.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 294.65 | 295.15 | 292.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 294.40 | 295.15 | 292.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 298.25 | 300.53 | 298.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 298.25 | 300.53 | 298.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 298.90 | 300.20 | 298.79 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 12:30:00 | 558.00 | 2024-05-31 15:15:00 | 613.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-29 13:30:00 | 557.95 | 2024-05-31 15:15:00 | 613.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-31 10:00:00 | 558.00 | 2024-05-31 15:15:00 | 613.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-31 10:45:00 | 558.70 | 2024-05-31 15:15:00 | 614.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-31 14:00:00 | 591.50 | 2024-06-03 09:15:00 | 650.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-31 15:00:00 | 599.40 | 2024-06-03 09:15:00 | 659.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 10:30:00 | 702.00 | 2024-06-19 09:15:00 | 674.00 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-06-18 11:00:00 | 694.50 | 2024-06-19 09:15:00 | 674.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2024-07-02 09:15:00 | 694.65 | 2024-07-02 10:15:00 | 685.45 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-07-08 13:30:00 | 719.00 | 2024-07-09 09:15:00 | 701.45 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-07-11 13:15:00 | 691.95 | 2024-07-12 09:15:00 | 700.95 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-11 14:00:00 | 691.00 | 2024-07-12 09:15:00 | 700.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-11 14:45:00 | 690.90 | 2024-07-12 09:15:00 | 700.95 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-07-22 14:00:00 | 636.00 | 2024-07-22 15:15:00 | 642.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-07-23 12:15:00 | 598.00 | 2024-07-24 09:15:00 | 642.70 | STOP_HIT | 1.00 | -7.47% |
| SELL | retest2 | 2024-07-24 09:30:00 | 632.95 | 2024-07-24 10:15:00 | 643.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-07-24 12:15:00 | 635.75 | 2024-07-24 14:15:00 | 644.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-29 12:00:00 | 615.60 | 2024-08-02 09:15:00 | 584.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 14:00:00 | 616.00 | 2024-08-02 09:15:00 | 585.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 14:30:00 | 616.70 | 2024-08-02 09:15:00 | 585.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 12:00:00 | 615.60 | 2024-08-02 10:15:00 | 609.50 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2024-07-30 14:00:00 | 616.00 | 2024-08-02 10:15:00 | 609.50 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2024-07-30 14:30:00 | 616.70 | 2024-08-02 10:15:00 | 609.50 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2024-08-20 14:15:00 | 539.00 | 2024-08-21 13:15:00 | 566.95 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2024-08-20 15:00:00 | 540.00 | 2024-08-21 13:15:00 | 566.95 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2024-08-21 10:45:00 | 541.30 | 2024-08-21 13:15:00 | 566.95 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest1 | 2024-08-29 09:15:00 | 566.00 | 2024-08-29 14:15:00 | 556.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-08-29 12:15:00 | 568.15 | 2024-08-29 14:15:00 | 556.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-08-30 10:00:00 | 568.65 | 2024-09-02 10:15:00 | 556.90 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-09-04 13:00:00 | 551.70 | 2024-09-09 09:15:00 | 524.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 14:15:00 | 551.05 | 2024-09-09 09:15:00 | 523.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 11:00:00 | 551.95 | 2024-09-09 09:15:00 | 524.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 15:00:00 | 550.15 | 2024-09-09 09:15:00 | 522.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 13:00:00 | 551.70 | 2024-09-10 09:15:00 | 496.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-04 14:15:00 | 551.05 | 2024-09-10 09:15:00 | 495.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-05 11:00:00 | 551.95 | 2024-09-10 09:15:00 | 496.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-05 15:00:00 | 550.15 | 2024-09-10 09:15:00 | 495.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-09 09:15:00 | 522.15 | 2024-09-10 09:15:00 | 496.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-09 09:15:00 | 522.15 | 2024-09-10 14:15:00 | 511.50 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2024-10-01 10:15:00 | 513.90 | 2024-10-04 09:15:00 | 488.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 513.90 | 2024-10-04 09:15:00 | 504.55 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2024-10-01 11:45:00 | 514.65 | 2024-10-04 09:15:00 | 488.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:45:00 | 514.65 | 2024-10-04 09:15:00 | 504.55 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2024-10-14 10:30:00 | 526.15 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-10-14 12:15:00 | 522.70 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-10-15 13:15:00 | 518.95 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-10-15 15:15:00 | 519.40 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-16 15:15:00 | 517.00 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-10-17 10:15:00 | 516.50 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-10-18 10:00:00 | 517.10 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-10-21 12:15:00 | 516.90 | 2024-10-21 12:15:00 | 516.55 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-10-29 10:30:00 | 462.80 | 2024-10-30 09:15:00 | 481.55 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-10-29 11:45:00 | 461.75 | 2024-10-30 09:15:00 | 481.55 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-11-19 14:00:00 | 446.25 | 2024-11-22 09:15:00 | 423.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 14:00:00 | 446.25 | 2024-11-25 09:15:00 | 446.60 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2024-11-25 09:45:00 | 445.85 | 2024-11-25 11:15:00 | 444.90 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-11-25 10:30:00 | 446.45 | 2024-11-25 11:15:00 | 444.90 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-12-04 09:15:00 | 508.00 | 2024-12-05 10:15:00 | 487.95 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-12-04 13:15:00 | 494.50 | 2024-12-05 10:15:00 | 487.95 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-12-04 14:30:00 | 492.20 | 2024-12-05 10:15:00 | 487.95 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-10 14:45:00 | 496.65 | 2024-12-11 10:15:00 | 546.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 09:15:00 | 531.95 | 2024-12-27 10:15:00 | 505.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 15:15:00 | 534.10 | 2024-12-27 10:15:00 | 507.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 09:15:00 | 531.95 | 2024-12-31 13:15:00 | 506.00 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2024-12-20 15:15:00 | 534.10 | 2024-12-31 13:15:00 | 506.00 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2025-01-08 09:15:00 | 481.25 | 2025-01-10 09:15:00 | 457.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 481.25 | 2025-01-13 11:15:00 | 433.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 474.20 | 2025-01-24 12:15:00 | 450.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 473.80 | 2025-01-24 12:15:00 | 450.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 474.20 | 2025-01-24 13:15:00 | 426.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 473.80 | 2025-01-24 13:15:00 | 426.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 12:15:00 | 352.85 | 2025-02-10 09:15:00 | 335.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:15:00 | 352.85 | 2025-02-11 09:15:00 | 317.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-12 11:15:00 | 294.50 | 2025-03-18 13:15:00 | 298.95 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-03-12 15:15:00 | 295.00 | 2025-03-18 13:15:00 | 298.95 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-03-13 10:30:00 | 295.95 | 2025-03-18 13:15:00 | 298.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-03-13 11:30:00 | 294.75 | 2025-03-18 13:15:00 | 298.95 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-01 09:30:00 | 372.80 | 2025-04-02 09:15:00 | 364.50 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-04-01 15:15:00 | 372.10 | 2025-04-02 09:15:00 | 364.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-04-03 09:15:00 | 373.50 | 2025-04-04 09:15:00 | 362.40 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-04-03 10:15:00 | 372.35 | 2025-04-04 09:15:00 | 362.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-04-07 09:15:00 | 342.95 | 2025-04-07 09:15:00 | 325.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 342.95 | 2025-04-07 13:15:00 | 358.55 | STOP_HIT | 0.50 | -4.55% |
| BUY | retest2 | 2025-04-21 10:15:00 | 390.65 | 2025-04-23 09:15:00 | 376.60 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-04-21 11:00:00 | 390.55 | 2025-04-23 09:15:00 | 376.60 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-04-22 09:15:00 | 394.80 | 2025-04-23 09:15:00 | 376.60 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2025-04-22 13:00:00 | 391.00 | 2025-04-23 09:15:00 | 376.60 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-04-30 09:15:00 | 361.45 | 2025-05-06 14:15:00 | 343.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 361.45 | 2025-05-07 15:15:00 | 343.15 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2025-06-03 09:45:00 | 396.25 | 2025-06-04 11:15:00 | 412.80 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-06-18 11:45:00 | 384.25 | 2025-06-19 12:15:00 | 365.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 384.45 | 2025-06-19 12:15:00 | 365.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:45:00 | 384.25 | 2025-06-20 10:15:00 | 373.35 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2025-06-18 12:15:00 | 384.45 | 2025-06-20 10:15:00 | 373.35 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-07-07 12:30:00 | 379.70 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.90% |
| SELL | retest2 | 2025-07-09 11:30:00 | 378.85 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-07-10 09:30:00 | 378.75 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-07-10 10:15:00 | 379.00 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2025-07-10 12:00:00 | 375.75 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-07-21 13:30:00 | 369.30 | 2025-07-28 09:15:00 | 350.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:30:00 | 369.35 | 2025-07-28 09:15:00 | 350.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 369.00 | 2025-07-28 09:15:00 | 350.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:30:00 | 369.30 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.27% |
| SELL | retest2 | 2025-07-22 09:30:00 | 369.35 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2025-07-22 10:15:00 | 369.00 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-08-08 09:30:00 | 337.25 | 2025-08-13 09:15:00 | 320.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:30:00 | 337.25 | 2025-08-18 09:15:00 | 325.10 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2025-09-19 11:15:00 | 343.70 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-19 12:00:00 | 343.80 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-19 13:00:00 | 343.75 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-08 12:30:00 | 339.90 | 2025-10-08 14:15:00 | 336.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-13 09:15:00 | 331.50 | 2025-10-23 09:15:00 | 323.70 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-11-04 11:15:00 | 322.30 | 2025-11-11 14:15:00 | 306.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 322.30 | 2025-11-11 14:15:00 | 313.75 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-12-01 11:15:00 | 286.50 | 2025-12-05 09:15:00 | 272.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:15:00 | 286.50 | 2025-12-08 10:15:00 | 257.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 292.40 | 2026-01-14 11:15:00 | 326.20 | STOP_HIT | 1.00 | -11.56% |
| SELL | retest2 | 2026-01-23 15:00:00 | 305.00 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-01-27 10:45:00 | 303.55 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-01-27 13:45:00 | 305.15 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-02-04 09:15:00 | 307.30 | 2026-02-09 14:15:00 | 309.05 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-18 09:30:00 | 289.05 | 2026-02-25 15:15:00 | 274.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 288.50 | 2026-02-25 15:15:00 | 274.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 10:45:00 | 288.90 | 2026-02-25 15:15:00 | 274.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:00:00 | 289.25 | 2026-02-25 15:15:00 | 274.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:30:00 | 289.05 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-02-19 15:15:00 | 288.50 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2026-02-20 10:45:00 | 288.90 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-02-20 15:00:00 | 289.25 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-02-23 10:15:00 | 287.00 | 2026-02-27 09:15:00 | 272.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 287.25 | 2026-02-27 09:15:00 | 272.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 287.30 | 2026-02-27 09:15:00 | 272.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 285.70 | 2026-02-27 09:15:00 | 271.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:30:00 | 284.30 | 2026-02-27 09:15:00 | 270.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 15:15:00 | 283.40 | 2026-02-27 09:15:00 | 269.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 287.00 | 2026-03-02 09:15:00 | 258.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 287.25 | 2026-03-02 09:15:00 | 258.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 287.30 | 2026-03-02 09:15:00 | 258.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 285.70 | 2026-03-02 09:15:00 | 257.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 11:30:00 | 284.30 | 2026-03-02 09:15:00 | 255.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 15:15:00 | 283.40 | 2026-03-02 09:15:00 | 255.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 263.30 | 2026-03-23 10:15:00 | 250.75 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-03-20 13:30:00 | 263.95 | 2026-03-23 12:15:00 | 250.13 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-03-20 12:15:00 | 263.30 | 2026-03-24 11:15:00 | 256.20 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2026-03-20 13:30:00 | 263.95 | 2026-03-24 11:15:00 | 256.20 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2026-04-10 09:15:00 | 272.94 | 2026-04-10 15:15:00 | 268.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-21 09:15:00 | 281.44 | 2026-04-24 10:15:00 | 278.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-04-24 10:00:00 | 282.00 | 2026-04-24 10:15:00 | 278.55 | STOP_HIT | 1.00 | -1.22% |
