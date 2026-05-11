# Sai Life Sciences Ltd. (SAILIFE)

## Backtest Summary

- **Window:** 2024-12-18 09:15:00 → 2026-05-08 15:15:00 (2396 bars)
- **Last close:** 1117.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 107 |
| ALERT1 | 74 |
| ALERT2 | 70 |
| ALERT2_SKIP | 35 |
| ALERT3 | 174 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 94 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 94 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 36 / 71
- **Target hits / Stop hits / Partials:** 2 / 94 / 11
- **Avg / median % per leg:** 0.01% / -1.09%
- **Sum % (uncompounded):** 1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 12 | 23.1% | 1 | 51 | 0 | -0.79% | -40.9% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.90% | -9.5% |
| BUY @ 3rd Alert (retest2) | 47 | 12 | 25.5% | 1 | 46 | 0 | -0.67% | -31.4% |
| SELL (all) | 55 | 24 | 43.6% | 1 | 43 | 11 | 0.77% | 42.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 24 | 43.6% | 1 | 43 | 11 | 0.77% | 42.5% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.90% | -9.5% |
| retest2 (combined) | 102 | 36 | 35.3% | 2 | 89 | 11 | 0.11% | 11.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 770.45 | 734.85 | 730.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 778.00 | 747.89 | 737.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 748.30 | 752.70 | 743.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 748.30 | 752.70 | 743.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 748.30 | 752.70 | 743.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 748.30 | 752.70 | 743.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 744.00 | 750.96 | 743.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 752.30 | 750.96 | 743.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 730.80 | 746.93 | 742.78 | SL hit (close<static) qty=1.00 sl=736.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 726.00 | 739.21 | 739.77 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 757.25 | 739.55 | 738.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 768.50 | 754.32 | 748.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 14:15:00 | 758.10 | 761.83 | 754.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 14:15:00 | 758.10 | 761.83 | 754.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 758.10 | 761.83 | 754.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:45:00 | 766.20 | 760.91 | 756.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 770.05 | 761.83 | 757.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 748.15 | 771.85 | 768.20 | SL hit (close<static) qty=1.00 sl=754.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 722.30 | 761.94 | 764.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 712.90 | 727.18 | 731.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 717.00 | 710.69 | 716.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 717.00 | 710.69 | 716.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 717.00 | 710.69 | 716.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 717.00 | 710.69 | 716.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 726.00 | 713.75 | 717.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:30:00 | 721.70 | 713.75 | 717.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 729.20 | 716.84 | 718.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 729.20 | 716.84 | 718.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 728.50 | 721.05 | 720.16 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 704.05 | 717.23 | 718.84 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 723.25 | 717.85 | 717.58 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 715.30 | 717.82 | 717.94 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 12:15:00 | 719.45 | 718.15 | 718.08 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 716.55 | 717.88 | 717.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 15:15:00 | 714.20 | 717.14 | 717.63 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 726.15 | 718.49 | 718.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 732.75 | 721.89 | 719.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 723.15 | 725.34 | 722.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 723.15 | 725.34 | 722.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 723.15 | 725.34 | 722.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 723.15 | 725.34 | 722.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 713.35 | 722.94 | 721.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 714.55 | 722.94 | 721.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 709.10 | 720.17 | 720.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 707.15 | 717.57 | 719.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 703.40 | 698.35 | 705.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 703.40 | 698.35 | 705.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 703.40 | 698.35 | 705.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 697.05 | 698.75 | 704.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 697.50 | 698.75 | 704.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 697.10 | 699.38 | 703.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 695.10 | 699.38 | 703.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 662.20 | 676.36 | 687.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 662.62 | 676.36 | 687.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 662.25 | 676.36 | 687.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 660.35 | 676.36 | 687.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 658.95 | 658.83 | 669.41 | SL hit (close>ema200) qty=0.50 sl=658.83 alert=retest2 |

### Cycle 13 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 675.00 | 670.23 | 669.81 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 11:15:00 | 666.05 | 669.24 | 669.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 13:15:00 | 662.35 | 667.09 | 668.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 11:15:00 | 661.60 | 656.89 | 660.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 11:15:00 | 661.60 | 656.89 | 660.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 661.60 | 656.89 | 660.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 661.60 | 656.89 | 660.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 666.90 | 658.89 | 660.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 662.55 | 658.89 | 660.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 13:15:00 | 685.35 | 664.18 | 663.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 692.40 | 675.86 | 669.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 14:15:00 | 678.00 | 680.05 | 674.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 678.00 | 680.05 | 674.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 703.60 | 694.67 | 686.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 720.25 | 694.67 | 686.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 713.20 | 726.74 | 709.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 703.60 | 722.13 | 724.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 703.60 | 722.13 | 724.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 698.80 | 717.47 | 721.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 684.00 | 678.84 | 690.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 692.85 | 678.84 | 690.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 683.85 | 679.84 | 690.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 677.50 | 679.84 | 690.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:45:00 | 678.60 | 679.98 | 689.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:45:00 | 680.00 | 680.48 | 687.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 679.45 | 680.27 | 686.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 669.60 | 678.41 | 684.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 663.40 | 675.72 | 682.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 663.25 | 675.72 | 682.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 663.35 | 673.25 | 681.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 643.62 | 664.90 | 675.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 644.67 | 664.90 | 675.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 646.00 | 664.90 | 675.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 645.48 | 664.90 | 675.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 659.80 | 663.88 | 674.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 664.50 | 661.98 | 671.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 664.50 | 661.98 | 671.44 | SL hit (close>ema200) qty=0.50 sl=661.98 alert=retest2 |

### Cycle 17 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 681.25 | 669.60 | 668.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 682.90 | 672.26 | 669.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 14:15:00 | 673.20 | 673.25 | 670.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 14:15:00 | 673.20 | 673.25 | 670.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 673.20 | 673.25 | 670.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:30:00 | 665.05 | 673.25 | 670.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 670.00 | 672.60 | 670.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 681.75 | 672.60 | 670.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 692.00 | 676.48 | 672.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:30:00 | 712.95 | 688.81 | 680.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 709.00 | 704.94 | 695.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 720.25 | 701.02 | 698.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 13:15:00 | 698.55 | 702.71 | 702.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 698.55 | 702.71 | 702.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 688.00 | 697.56 | 700.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 699.30 | 692.97 | 695.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 699.30 | 692.97 | 695.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 699.30 | 692.97 | 695.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 699.20 | 692.97 | 695.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 702.30 | 694.84 | 695.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 699.20 | 694.84 | 695.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 706.65 | 697.20 | 696.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 721.95 | 702.77 | 699.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 725.50 | 726.84 | 718.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:15:00 | 740.35 | 726.84 | 718.98 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:15:00 | 733.45 | 727.72 | 720.10 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:00:00 | 732.95 | 728.77 | 721.26 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 719.00 | 727.01 | 721.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-06 12:15:00 | 719.00 | 727.01 | 721.78 | SL hit (close<ema400) qty=1.00 sl=721.78 alert=retest1 |

### Cycle 20 — SELL (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 14:15:00 | 697.70 | 718.20 | 718.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 13:15:00 | 688.00 | 700.57 | 708.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 11:15:00 | 697.40 | 693.24 | 700.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:45:00 | 699.50 | 693.24 | 700.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 699.75 | 695.74 | 700.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 698.00 | 695.74 | 700.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 695.70 | 695.73 | 699.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 683.30 | 695.73 | 699.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 15:15:00 | 707.50 | 700.38 | 700.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 707.50 | 700.38 | 700.26 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 698.75 | 700.06 | 700.12 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 703.60 | 700.76 | 700.44 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 15:15:00 | 697.00 | 700.20 | 700.33 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 705.80 | 701.30 | 700.81 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 696.85 | 700.15 | 700.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 684.25 | 696.20 | 698.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 704.70 | 693.07 | 695.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 704.70 | 693.07 | 695.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 704.70 | 693.07 | 695.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 705.00 | 693.07 | 695.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 705.00 | 695.46 | 696.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 698.75 | 695.46 | 696.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 699.55 | 695.35 | 695.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 699.55 | 695.35 | 695.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 13:15:00 | 710.00 | 699.35 | 697.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 722.05 | 722.88 | 716.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:15:00 | 734.80 | 722.88 | 716.30 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 727.65 | 733.41 | 727.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 727.65 | 733.41 | 727.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 725.00 | 731.73 | 727.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 725.00 | 731.73 | 727.27 | SL hit (close<ema400) qty=1.00 sl=727.27 alert=retest1 |

### Cycle 28 — SELL (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 12:15:00 | 745.95 | 750.54 | 750.59 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 765.70 | 751.43 | 750.73 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 727.30 | 748.00 | 750.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 713.20 | 734.48 | 743.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 680.55 | 678.87 | 700.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:30:00 | 682.95 | 678.87 | 700.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 657.60 | 654.16 | 669.33 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 716.95 | 675.01 | 673.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 725.40 | 685.08 | 677.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 739.55 | 739.72 | 720.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 739.55 | 739.72 | 720.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 720.30 | 732.77 | 722.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 732.55 | 729.40 | 722.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 730.50 | 729.40 | 722.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 727.75 | 729.37 | 723.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:45:00 | 729.05 | 725.72 | 723.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 722.10 | 725.00 | 723.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:45:00 | 724.00 | 725.00 | 723.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 725.45 | 725.09 | 723.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 722.50 | 725.09 | 723.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 723.15 | 724.70 | 723.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:00:00 | 723.15 | 724.70 | 723.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 721.85 | 724.13 | 723.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 721.85 | 724.13 | 723.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 721.05 | 723.52 | 723.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 722.20 | 723.52 | 723.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 724.50 | 723.71 | 723.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-22 11:15:00 | 719.60 | 722.58 | 722.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 11:15:00 | 719.60 | 722.58 | 722.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 12:15:00 | 709.95 | 720.05 | 721.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 713.40 | 712.88 | 716.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 11:00:00 | 713.40 | 712.88 | 716.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 714.20 | 713.20 | 716.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:45:00 | 715.90 | 713.20 | 716.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 717.45 | 714.05 | 716.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:45:00 | 717.00 | 714.05 | 716.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 712.15 | 713.67 | 716.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:45:00 | 717.75 | 713.67 | 716.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 714.00 | 713.47 | 715.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 714.00 | 713.47 | 715.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 713.20 | 713.41 | 715.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 714.45 | 713.41 | 715.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 689.15 | 684.81 | 691.09 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 719.90 | 695.05 | 694.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 723.50 | 713.69 | 708.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 15:15:00 | 717.60 | 717.95 | 713.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:15:00 | 722.50 | 717.95 | 713.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 720.50 | 718.46 | 714.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 727.00 | 721.86 | 716.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 709.05 | 719.91 | 717.91 | SL hit (close<static) qty=1.00 sl=712.80 alert=retest2 |

### Cycle 34 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 704.00 | 717.87 | 718.75 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 719.70 | 712.78 | 712.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 720.90 | 714.72 | 713.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 714.55 | 714.69 | 713.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 714.55 | 714.69 | 713.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 711.50 | 714.05 | 713.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 711.50 | 714.05 | 713.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 726.20 | 716.48 | 714.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 728.00 | 716.48 | 714.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 736.65 | 746.50 | 747.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 736.65 | 746.50 | 747.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 730.30 | 739.37 | 743.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 15:15:00 | 738.00 | 736.08 | 740.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:15:00 | 743.00 | 736.08 | 740.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 745.55 | 737.97 | 741.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 748.55 | 737.97 | 741.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 745.30 | 739.44 | 741.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 739.35 | 739.42 | 741.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 737.95 | 730.12 | 729.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 737.95 | 730.12 | 729.68 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 729.50 | 730.69 | 730.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 727.70 | 729.90 | 730.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 718.10 | 717.35 | 722.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 718.10 | 717.35 | 722.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 714.00 | 716.68 | 721.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 722.55 | 718.20 | 721.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 724.25 | 719.41 | 722.14 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 743.75 | 727.35 | 725.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 776.10 | 737.55 | 730.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 730.25 | 737.92 | 731.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 730.25 | 737.92 | 731.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 730.25 | 737.92 | 731.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 728.00 | 737.92 | 731.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 732.25 | 736.79 | 732.00 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 724.45 | 729.51 | 730.06 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 739.35 | 729.90 | 729.58 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 726.45 | 729.04 | 729.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 725.00 | 727.14 | 728.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 734.85 | 728.68 | 728.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 734.85 | 728.68 | 728.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 734.85 | 728.68 | 728.78 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 733.75 | 729.69 | 729.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 761.45 | 739.42 | 735.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 762.95 | 764.85 | 759.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 14:00:00 | 762.95 | 764.85 | 759.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 766.60 | 768.42 | 763.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:30:00 | 764.25 | 768.42 | 763.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 760.65 | 766.87 | 763.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 760.65 | 766.87 | 763.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 752.15 | 763.92 | 762.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 752.15 | 763.92 | 762.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 751.75 | 761.49 | 761.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 751.75 | 761.49 | 761.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 750.25 | 759.24 | 760.18 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 767.25 | 760.88 | 760.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 14:15:00 | 768.85 | 762.47 | 761.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 11:15:00 | 761.80 | 764.00 | 762.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 761.80 | 764.00 | 762.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 761.80 | 764.00 | 762.51 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 756.35 | 761.14 | 761.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 752.15 | 759.34 | 760.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 737.80 | 732.69 | 739.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 737.80 | 732.69 | 739.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 737.80 | 732.69 | 739.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 737.80 | 732.69 | 739.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 742.40 | 734.63 | 739.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 737.20 | 734.63 | 739.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 741.00 | 735.91 | 739.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 741.00 | 735.91 | 739.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 732.15 | 735.15 | 739.09 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 767.55 | 742.79 | 741.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 782.00 | 750.63 | 745.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 10:15:00 | 750.70 | 751.68 | 747.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 11:00:00 | 750.70 | 751.68 | 747.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 738.50 | 750.34 | 748.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 738.50 | 750.34 | 748.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 750.65 | 750.40 | 748.96 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 741.15 | 747.13 | 747.63 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 751.40 | 747.99 | 747.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 746.80 | 747.75 | 747.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 15:15:00 | 744.00 | 747.00 | 747.52 | Break + close below crossover candle low |

### Cycle 51 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 767.85 | 751.17 | 749.37 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 14:15:00 | 747.50 | 753.28 | 753.38 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 766.45 | 755.06 | 754.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 10:15:00 | 772.60 | 762.01 | 757.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 766.90 | 769.24 | 764.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 766.90 | 769.24 | 764.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 768.55 | 769.11 | 764.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 767.55 | 769.11 | 764.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 830.00 | 785.05 | 774.29 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 780.60 | 790.71 | 790.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 779.00 | 784.37 | 786.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 784.85 | 784.47 | 786.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 784.85 | 784.47 | 786.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 784.85 | 784.47 | 786.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 15:00:00 | 781.40 | 784.17 | 785.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:00:00 | 781.50 | 783.77 | 785.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:00:00 | 782.80 | 782.89 | 784.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 793.15 | 784.32 | 784.50 | SL hit (close>static) qty=1.00 sl=791.85 alert=retest2 |

### Cycle 55 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 796.60 | 786.78 | 785.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 817.95 | 793.01 | 788.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 811.25 | 811.82 | 802.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 811.25 | 811.82 | 802.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 810.95 | 811.24 | 805.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 815.90 | 811.89 | 808.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 806.00 | 815.80 | 816.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 806.00 | 815.80 | 816.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 804.35 | 812.24 | 814.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 808.85 | 807.70 | 811.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 808.85 | 807.70 | 811.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 808.85 | 807.70 | 811.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 810.00 | 807.70 | 811.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 810.30 | 808.54 | 811.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 809.45 | 808.54 | 811.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 807.90 | 808.41 | 810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 808.70 | 808.41 | 810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 806.25 | 807.77 | 810.11 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 816.25 | 811.24 | 811.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 819.00 | 813.14 | 812.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 835.30 | 836.02 | 829.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 835.30 | 836.02 | 829.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 824.60 | 833.73 | 829.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 824.60 | 833.73 | 829.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 824.60 | 831.91 | 828.61 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 823.00 | 826.80 | 826.93 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 828.15 | 827.07 | 827.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 830.90 | 827.84 | 827.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 15:15:00 | 828.25 | 828.51 | 827.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:15:00 | 840.00 | 828.51 | 827.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 828.25 | 838.95 | 835.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 828.25 | 838.95 | 835.53 | SL hit (close<ema400) qty=1.00 sl=835.53 alert=retest1 |

### Cycle 60 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 819.00 | 831.87 | 832.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 810.20 | 823.75 | 828.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 804.25 | 799.02 | 808.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 804.25 | 799.02 | 808.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 804.00 | 800.02 | 808.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 806.80 | 800.02 | 808.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 805.20 | 801.88 | 807.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 805.20 | 801.88 | 807.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 790.70 | 800.19 | 805.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:00:00 | 789.35 | 793.24 | 797.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:15:00 | 789.60 | 792.61 | 797.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 788.35 | 792.84 | 796.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 789.30 | 790.82 | 795.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 791.20 | 789.85 | 792.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 791.20 | 789.85 | 792.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 825.00 | 797.21 | 795.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 825.00 | 797.21 | 795.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 888.75 | 846.85 | 827.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 877.05 | 877.09 | 857.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:45:00 | 876.55 | 877.09 | 857.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 888.00 | 892.37 | 885.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 899.00 | 892.37 | 885.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 898.40 | 893.57 | 886.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 913.80 | 903.07 | 896.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 912.85 | 907.98 | 900.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 911.10 | 907.13 | 901.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:00:00 | 911.65 | 906.40 | 902.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 926.10 | 911.40 | 905.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 927.90 | 914.62 | 907.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 927.50 | 914.62 | 907.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:45:00 | 927.90 | 921.35 | 913.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 931.30 | 924.48 | 916.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 923.95 | 927.77 | 922.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 924.15 | 927.33 | 922.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 921.20 | 926.10 | 922.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 920.20 | 926.10 | 922.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 918.00 | 924.48 | 922.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:00:00 | 923.75 | 924.33 | 922.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 906.60 | 919.51 | 920.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 906.60 | 919.51 | 920.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 867.95 | 907.02 | 914.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 13:15:00 | 830.80 | 830.79 | 848.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:45:00 | 829.80 | 830.79 | 848.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 844.25 | 835.72 | 842.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 843.55 | 835.72 | 842.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 845.00 | 837.57 | 843.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 845.00 | 837.57 | 843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 844.00 | 838.86 | 843.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 844.45 | 838.86 | 843.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 847.00 | 840.49 | 843.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 834.80 | 840.49 | 843.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 838.50 | 831.87 | 835.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 838.50 | 831.87 | 835.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 843.10 | 834.12 | 836.50 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 851.15 | 840.52 | 839.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 859.20 | 851.99 | 848.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 851.00 | 856.55 | 853.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 851.00 | 856.55 | 853.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 851.00 | 856.55 | 853.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 851.00 | 856.55 | 853.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 854.40 | 856.12 | 853.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 855.35 | 855.60 | 853.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 855.70 | 854.68 | 853.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:15:00 | 856.00 | 854.68 | 853.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 854.60 | 854.91 | 853.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 857.00 | 855.70 | 854.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 863.00 | 857.42 | 855.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 880.25 | 893.52 | 895.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 880.25 | 893.52 | 895.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 874.15 | 889.65 | 893.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 10:15:00 | 893.50 | 885.69 | 889.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 10:15:00 | 893.50 | 885.69 | 889.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 893.50 | 885.69 | 889.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 893.50 | 885.69 | 889.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 897.50 | 888.05 | 890.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 897.30 | 888.05 | 890.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 895.30 | 889.50 | 890.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 893.05 | 889.50 | 890.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:00:00 | 894.50 | 890.50 | 891.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 893.30 | 891.06 | 891.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 895.70 | 891.96 | 891.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 895.70 | 891.96 | 891.72 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 885.65 | 892.50 | 892.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 854.60 | 879.85 | 885.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 863.35 | 861.86 | 871.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 863.35 | 861.86 | 871.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 863.35 | 861.86 | 871.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 861.00 | 862.17 | 870.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 886.75 | 868.91 | 871.28 | SL hit (close>static) qty=1.00 sl=874.25 alert=retest2 |

### Cycle 67 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 877.65 | 869.76 | 869.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 883.15 | 873.69 | 871.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 879.45 | 882.95 | 878.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 879.45 | 882.95 | 878.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 879.45 | 882.95 | 878.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 879.45 | 882.95 | 878.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 863.45 | 878.58 | 877.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 860.95 | 878.58 | 877.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 861.00 | 875.06 | 876.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 859.85 | 864.85 | 868.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 851.50 | 850.35 | 856.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:30:00 | 851.20 | 850.35 | 856.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 846.10 | 849.69 | 854.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 844.90 | 849.69 | 854.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 862.30 | 852.91 | 855.09 | SL hit (close>static) qty=1.00 sl=859.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 862.90 | 856.63 | 856.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 866.45 | 858.49 | 857.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 865.05 | 866.36 | 862.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 863.55 | 866.36 | 862.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 860.05 | 865.10 | 862.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 860.05 | 865.10 | 862.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 862.80 | 864.64 | 862.44 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 854.00 | 861.25 | 861.63 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 866.95 | 861.48 | 860.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 873.85 | 863.96 | 862.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 868.50 | 873.69 | 869.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 868.50 | 873.69 | 869.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 868.50 | 873.69 | 869.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 868.50 | 873.69 | 869.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 869.30 | 872.81 | 869.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 868.80 | 872.81 | 869.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 871.35 | 872.52 | 869.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 868.05 | 872.52 | 869.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 865.00 | 871.25 | 869.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 865.00 | 871.25 | 869.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 866.05 | 870.21 | 869.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 871.90 | 870.95 | 870.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 879.40 | 884.00 | 884.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 879.40 | 884.00 | 884.40 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 894.00 | 884.90 | 884.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 906.85 | 895.48 | 890.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 914.60 | 915.88 | 908.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:45:00 | 914.75 | 915.88 | 908.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 924.00 | 920.37 | 915.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 935.45 | 921.37 | 918.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 903.65 | 919.35 | 920.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 903.65 | 919.35 | 920.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 13:15:00 | 882.90 | 899.53 | 909.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 884.60 | 884.47 | 894.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 13:15:00 | 884.60 | 884.47 | 894.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 884.60 | 884.47 | 894.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:45:00 | 884.75 | 884.47 | 894.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 903.80 | 888.34 | 895.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 903.80 | 888.34 | 895.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 910.00 | 892.67 | 897.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 901.40 | 894.14 | 897.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 883.00 | 881.70 | 881.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 883.00 | 881.70 | 881.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 886.85 | 882.73 | 882.08 | Break + close above crossover candle high |

### Cycle 76 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 875.00 | 881.32 | 881.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 863.25 | 877.71 | 879.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 866.75 | 865.95 | 869.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 866.75 | 865.95 | 869.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 866.75 | 865.95 | 869.36 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 874.95 | 870.59 | 870.46 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 866.55 | 869.78 | 870.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 860.85 | 867.99 | 869.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 865.95 | 864.22 | 866.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 865.95 | 864.22 | 866.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 865.95 | 864.22 | 866.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 865.95 | 864.22 | 866.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 865.45 | 864.46 | 866.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 865.15 | 864.46 | 866.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 886.95 | 868.79 | 867.68 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 876.50 | 878.42 | 878.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 868.85 | 875.31 | 876.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 874.15 | 872.48 | 874.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 874.15 | 872.48 | 874.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 874.15 | 872.48 | 874.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 874.75 | 872.48 | 874.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 876.80 | 873.35 | 874.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 876.80 | 873.35 | 874.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 872.20 | 873.12 | 874.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 876.75 | 873.12 | 874.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 871.00 | 869.80 | 871.84 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 875.10 | 872.82 | 872.81 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 863.05 | 871.15 | 872.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 857.95 | 866.85 | 869.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 867.10 | 862.70 | 866.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 867.10 | 862.70 | 866.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 867.10 | 862.70 | 866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 867.10 | 862.70 | 866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 881.85 | 866.53 | 867.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 881.85 | 866.53 | 867.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 882.05 | 869.63 | 868.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 884.55 | 877.20 | 873.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 10:15:00 | 880.80 | 884.18 | 879.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 11:00:00 | 880.80 | 884.18 | 879.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 887.40 | 884.82 | 880.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 896.00 | 884.82 | 880.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 886.25 | 896.76 | 897.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 886.25 | 896.76 | 897.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 885.00 | 894.41 | 896.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 888.90 | 879.12 | 883.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 888.90 | 879.12 | 883.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 888.90 | 879.12 | 883.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 888.90 | 879.12 | 883.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 881.75 | 879.65 | 883.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 879.80 | 879.65 | 883.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 895.80 | 884.66 | 884.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 895.80 | 884.66 | 884.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 920.70 | 891.87 | 887.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 904.40 | 906.37 | 900.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:45:00 | 904.65 | 906.37 | 900.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 899.15 | 904.47 | 900.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 898.75 | 904.47 | 900.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 898.50 | 903.27 | 900.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 898.50 | 903.27 | 900.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 900.85 | 902.27 | 900.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 907.55 | 902.27 | 900.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 906.00 | 902.51 | 900.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 908.70 | 902.51 | 900.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 904.50 | 903.03 | 901.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 900.95 | 902.61 | 901.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 900.95 | 902.61 | 901.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 901.15 | 902.32 | 901.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 901.45 | 902.32 | 901.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 901.25 | 902.11 | 901.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 901.00 | 902.11 | 901.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 901.00 | 901.88 | 901.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 903.10 | 901.88 | 901.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 901.00 | 901.71 | 901.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 898.60 | 900.73 | 900.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 898.60 | 900.73 | 900.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 896.00 | 899.27 | 900.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 896.25 | 890.58 | 893.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 896.25 | 890.58 | 893.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 896.25 | 890.58 | 893.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 896.25 | 890.58 | 893.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 895.50 | 891.56 | 893.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 897.95 | 891.56 | 893.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 899.95 | 895.31 | 895.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 915.15 | 899.28 | 896.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 900.80 | 901.01 | 898.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 900.80 | 901.01 | 898.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 900.80 | 901.01 | 898.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 900.30 | 901.01 | 898.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 900.70 | 901.07 | 898.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 900.00 | 901.07 | 898.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 898.20 | 903.91 | 901.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 898.35 | 903.91 | 901.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 903.10 | 903.74 | 901.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 909.80 | 902.02 | 901.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 939.30 | 957.59 | 958.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 939.30 | 957.59 | 958.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 925.00 | 940.42 | 948.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 904.50 | 899.89 | 913.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:30:00 | 903.85 | 899.89 | 913.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 903.65 | 902.31 | 910.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 906.30 | 902.31 | 910.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 910.00 | 904.20 | 908.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 910.00 | 904.20 | 908.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 906.50 | 904.66 | 908.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 904.00 | 904.66 | 908.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 858.80 | 875.90 | 889.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 813.60 | 833.10 | 850.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 89 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 840.25 | 821.61 | 821.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 844.45 | 826.18 | 823.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 839.20 | 841.71 | 834.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 839.20 | 841.71 | 834.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 827.30 | 838.83 | 834.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 827.30 | 838.83 | 834.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 826.00 | 836.26 | 833.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 828.65 | 834.47 | 832.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 836.15 | 832.80 | 832.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 829.55 | 832.28 | 832.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 829.55 | 832.28 | 832.31 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 834.50 | 832.72 | 832.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 15:15:00 | 836.00 | 833.38 | 832.83 | Break + close above crossover candle high |

### Cycle 92 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 816.00 | 829.90 | 831.30 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 839.90 | 832.68 | 832.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 870.05 | 845.35 | 839.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 855.60 | 866.47 | 856.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 855.60 | 866.47 | 856.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 855.60 | 866.47 | 856.10 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 820.10 | 847.12 | 849.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 14:15:00 | 812.85 | 840.27 | 846.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 840.10 | 818.54 | 826.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 840.10 | 818.54 | 826.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 840.10 | 818.54 | 826.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 831.00 | 818.54 | 826.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 850.60 | 832.76 | 831.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 850.60 | 832.76 | 831.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 887.45 | 849.10 | 840.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 925.50 | 936.70 | 920.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:00:00 | 925.50 | 936.70 | 920.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 891.30 | 924.10 | 920.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 891.05 | 924.10 | 920.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 907.60 | 920.80 | 918.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 908.70 | 920.80 | 918.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 934.85 | 947.03 | 948.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 09:15:00 | 934.85 | 947.03 | 948.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 923.85 | 939.62 | 944.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 943.90 | 933.53 | 939.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 943.90 | 933.53 | 939.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 943.90 | 933.53 | 939.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 943.90 | 933.53 | 939.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 952.80 | 937.39 | 940.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 950.60 | 937.39 | 940.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 958.65 | 945.34 | 943.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 964.95 | 949.26 | 945.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 970.50 | 990.70 | 983.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 970.50 | 990.70 | 983.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 970.50 | 990.70 | 983.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 10:30:00 | 979.45 | 988.73 | 983.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-05 14:15:00 | 1077.40 | 1035.46 | 1013.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1012.60 | 1020.02 | 1020.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 1006.40 | 1015.73 | 1018.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 1017.00 | 1012.07 | 1014.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 15:15:00 | 1017.00 | 1012.07 | 1014.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1017.00 | 1012.07 | 1014.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1020.70 | 1012.07 | 1014.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1015.95 | 1012.85 | 1015.03 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 1018.70 | 1016.62 | 1016.38 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1008.60 | 1015.17 | 1015.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 996.05 | 1011.35 | 1013.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1006.90 | 1006.71 | 1010.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 1006.90 | 1006.71 | 1010.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1006.90 | 1006.71 | 1010.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 993.50 | 1007.09 | 1009.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 943.82 | 977.57 | 993.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 15:15:00 | 969.95 | 968.09 | 984.64 | SL hit (close>ema200) qty=0.50 sl=968.09 alert=retest2 |

### Cycle 101 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 986.00 | 980.70 | 979.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1002.75 | 985.11 | 982.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 1005.20 | 1006.86 | 999.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 1005.20 | 1006.86 | 999.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1010.85 | 1007.66 | 1000.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1010.85 | 1007.66 | 1000.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 999.75 | 1006.78 | 1003.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 999.75 | 1006.78 | 1003.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 998.70 | 1005.16 | 1003.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 992.20 | 1005.16 | 1003.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 973.00 | 998.73 | 1000.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 966.00 | 984.86 | 993.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 989.25 | 985.74 | 992.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 989.25 | 985.74 | 992.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 981.00 | 985.26 | 991.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 1009.25 | 985.26 | 991.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1001.20 | 988.45 | 992.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 998.15 | 988.45 | 992.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:15:00 | 998.55 | 992.47 | 993.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 1000.85 | 995.12 | 994.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 1000.85 | 995.12 | 994.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1020.05 | 1000.10 | 996.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 11:15:00 | 993.80 | 999.13 | 996.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 11:15:00 | 993.80 | 999.13 | 996.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 993.80 | 999.13 | 996.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:45:00 | 993.30 | 999.13 | 996.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 995.55 | 998.41 | 996.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:45:00 | 993.90 | 998.41 | 996.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 1010.85 | 1000.90 | 998.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:30:00 | 996.85 | 1000.90 | 998.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 985.10 | 1000.70 | 998.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 985.10 | 1000.70 | 998.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 992.20 | 999.00 | 998.32 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 986.90 | 996.58 | 997.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 984.30 | 992.27 | 995.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 979.60 | 979.34 | 986.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:45:00 | 979.00 | 979.34 | 986.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 977.85 | 979.04 | 985.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:30:00 | 981.80 | 979.04 | 985.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 983.80 | 977.28 | 982.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 977.55 | 977.28 | 982.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 09:15:00 | 928.67 | 944.41 | 953.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 943.65 | 943.63 | 951.07 | SL hit (close>ema200) qty=0.50 sl=943.63 alert=retest2 |

### Cycle 105 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 969.85 | 952.71 | 952.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 991.30 | 972.60 | 964.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 978.65 | 980.50 | 972.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 978.65 | 980.50 | 972.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 984.95 | 994.15 | 986.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 995.25 | 992.38 | 987.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 994.55 | 992.03 | 987.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1000.70 | 991.85 | 988.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 969.50 | 985.37 | 987.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 969.50 | 985.37 | 987.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 10:15:00 | 962.00 | 980.69 | 985.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 15:15:00 | 973.90 | 972.96 | 978.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:15:00 | 983.85 | 972.96 | 978.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 985.00 | 975.37 | 979.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:00:00 | 973.75 | 976.08 | 979.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:45:00 | 970.25 | 974.98 | 978.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 11:30:00 | 962.15 | 959.93 | 966.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 984.65 | 968.74 | 967.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 984.65 | 968.74 | 967.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 11:15:00 | 991.50 | 973.29 | 969.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 1029.90 | 1030.87 | 1011.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 09:45:00 | 1030.00 | 1030.87 | 1011.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1021.00 | 1028.10 | 1018.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1042.70 | 1028.10 | 1018.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 11:30:00 | 1031.45 | 1031.20 | 1022.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 1034.20 | 1031.20 | 1022.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-27 09:15:00 | 752.30 | 2024-12-27 09:15:00 | 730.80 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-01-02 12:45:00 | 766.20 | 2025-01-06 09:15:00 | 748.15 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-01-02 15:15:00 | 770.05 | 2025-01-06 09:15:00 | 748.15 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-01-23 11:30:00 | 697.05 | 2025-01-27 09:15:00 | 662.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 697.50 | 2025-01-27 09:15:00 | 662.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 697.10 | 2025-01-27 09:15:00 | 662.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 695.10 | 2025-01-27 09:15:00 | 660.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 697.05 | 2025-01-28 11:15:00 | 658.95 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2025-01-23 12:15:00 | 697.50 | 2025-01-28 11:15:00 | 658.95 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2025-01-23 14:30:00 | 697.10 | 2025-01-28 11:15:00 | 658.95 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2025-01-23 15:15:00 | 695.10 | 2025-01-28 11:15:00 | 658.95 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2025-01-29 11:30:00 | 673.05 | 2025-01-29 15:15:00 | 675.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-01-29 12:15:00 | 672.65 | 2025-01-29 15:15:00 | 675.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-01-29 12:45:00 | 671.25 | 2025-01-29 15:15:00 | 675.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-02-06 10:15:00 | 720.25 | 2025-02-11 09:15:00 | 703.60 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-02-07 09:30:00 | 713.20 | 2025-02-11 09:15:00 | 703.60 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-02-13 10:15:00 | 677.50 | 2025-02-14 13:15:00 | 643.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:45:00 | 678.60 | 2025-02-14 13:15:00 | 644.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:45:00 | 680.00 | 2025-02-14 13:15:00 | 646.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 679.45 | 2025-02-14 13:15:00 | 645.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:15:00 | 677.50 | 2025-02-17 09:15:00 | 664.50 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-02-13 10:45:00 | 678.60 | 2025-02-17 09:15:00 | 664.50 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-02-13 13:45:00 | 680.00 | 2025-02-17 09:15:00 | 664.50 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2025-02-13 15:00:00 | 679.45 | 2025-02-17 09:15:00 | 664.50 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-02-14 10:45:00 | 663.40 | 2025-02-18 14:15:00 | 673.15 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-02-14 11:15:00 | 663.25 | 2025-02-19 10:15:00 | 681.25 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-02-14 12:00:00 | 663.35 | 2025-02-19 10:15:00 | 681.25 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-02-14 15:00:00 | 659.80 | 2025-02-19 10:15:00 | 681.25 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-02-18 09:15:00 | 653.00 | 2025-02-19 10:15:00 | 681.25 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-02-20 14:30:00 | 712.95 | 2025-02-27 13:15:00 | 698.55 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-02-24 09:15:00 | 709.00 | 2025-02-27 13:15:00 | 698.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-02-25 09:15:00 | 720.25 | 2025-02-27 13:15:00 | 698.55 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2025-03-06 09:15:00 | 740.35 | 2025-03-06 12:15:00 | 719.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest1 | 2025-03-06 10:15:00 | 733.45 | 2025-03-06 12:15:00 | 719.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2025-03-06 11:00:00 | 732.95 | 2025-03-06 12:15:00 | 719.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-03-11 09:15:00 | 683.30 | 2025-03-11 15:15:00 | 707.50 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-03-18 09:15:00 | 698.75 | 2025-03-19 10:15:00 | 699.55 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-03-24 09:15:00 | 734.80 | 2025-03-25 11:15:00 | 725.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-25 15:15:00 | 738.25 | 2025-04-02 12:15:00 | 745.95 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-04-17 11:30:00 | 732.55 | 2025-04-22 11:15:00 | 719.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-04-17 12:15:00 | 730.50 | 2025-04-22 11:15:00 | 719.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-04-17 12:45:00 | 727.75 | 2025-04-22 11:15:00 | 719.60 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-04-21 10:45:00 | 729.05 | 2025-04-22 11:15:00 | 719.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-05-06 11:45:00 | 727.00 | 2025-05-07 09:15:00 | 709.05 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-05-08 09:30:00 | 724.70 | 2025-05-08 15:15:00 | 704.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-05-08 11:45:00 | 725.60 | 2025-05-08 15:15:00 | 704.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-05-13 15:15:00 | 728.00 | 2025-05-20 09:15:00 | 736.65 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-05-21 12:00:00 | 739.35 | 2025-05-26 14:15:00 | 737.95 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-07-10 15:00:00 | 781.40 | 2025-07-14 09:15:00 | 793.15 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-11 10:00:00 | 781.50 | 2025-07-14 09:15:00 | 793.15 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-11 15:00:00 | 782.80 | 2025-07-14 09:15:00 | 793.15 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-17 09:15:00 | 815.90 | 2025-07-21 13:15:00 | 806.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2025-07-30 09:15:00 | 840.00 | 2025-07-31 09:15:00 | 828.25 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-06 13:00:00 | 789.35 | 2025-08-08 09:15:00 | 825.00 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-08-06 14:15:00 | 789.60 | 2025-08-08 09:15:00 | 825.00 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-08-06 15:15:00 | 788.35 | 2025-08-08 09:15:00 | 825.00 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-08-07 09:30:00 | 789.30 | 2025-08-08 09:15:00 | 825.00 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-08-19 12:00:00 | 913.80 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-20 09:15:00 | 912.85 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-08-20 10:30:00 | 911.10 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-08-20 15:00:00 | 911.65 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-21 10:30:00 | 927.90 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-08-21 11:00:00 | 927.50 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-21 14:45:00 | 927.90 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-08-22 09:30:00 | 931.30 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-08-25 13:00:00 | 923.75 | 2025-08-25 14:15:00 | 906.60 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-09-09 09:30:00 | 855.35 | 2025-09-22 12:15:00 | 880.25 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-09-09 11:45:00 | 855.70 | 2025-09-22 12:15:00 | 880.25 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-09-09 12:15:00 | 856.00 | 2025-09-22 12:15:00 | 880.25 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-09-09 13:45:00 | 854.60 | 2025-09-22 12:15:00 | 880.25 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2025-09-10 09:30:00 | 863.00 | 2025-09-22 12:15:00 | 880.25 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2025-09-23 13:15:00 | 893.05 | 2025-09-24 09:15:00 | 895.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-09-23 14:00:00 | 894.50 | 2025-09-24 09:15:00 | 895.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-09-23 14:45:00 | 893.30 | 2025-09-24 09:15:00 | 895.70 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-29 10:30:00 | 861.00 | 2025-09-29 14:15:00 | 886.75 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-09-30 11:30:00 | 860.75 | 2025-10-01 09:15:00 | 876.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-09-30 13:30:00 | 861.40 | 2025-10-01 09:15:00 | 876.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-10-10 10:15:00 | 844.90 | 2025-10-10 12:15:00 | 862.30 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-10-20 12:45:00 | 871.90 | 2025-10-27 14:15:00 | 879.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-11-04 09:15:00 | 935.45 | 2025-11-07 09:15:00 | 903.65 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-11-11 09:45:00 | 901.40 | 2025-11-17 12:15:00 | 883.00 | STOP_HIT | 1.00 | 2.04% |
| BUY | retest2 | 2025-12-11 12:15:00 | 896.00 | 2025-12-17 09:15:00 | 886.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-19 11:15:00 | 879.80 | 2025-12-19 13:15:00 | 895.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-24 09:15:00 | 907.55 | 2025-12-26 11:15:00 | 898.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-24 09:45:00 | 906.00 | 2025-12-26 11:15:00 | 898.60 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-24 10:15:00 | 908.70 | 2025-12-26 11:15:00 | 898.60 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-24 11:45:00 | 904.50 | 2025-12-26 11:15:00 | 898.60 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-01-02 09:15:00 | 909.80 | 2026-01-09 09:15:00 | 939.30 | STOP_HIT | 1.00 | 3.24% |
| SELL | retest2 | 2026-01-14 15:15:00 | 904.00 | 2026-01-19 09:15:00 | 858.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 904.00 | 2026-01-21 09:15:00 | 813.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-29 14:45:00 | 828.65 | 2026-01-30 13:15:00 | 829.55 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-01-30 09:30:00 | 836.15 | 2026-01-30 13:15:00 | 829.55 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-02-06 10:15:00 | 831.00 | 2026-02-06 12:15:00 | 850.60 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-02-13 11:15:00 | 908.70 | 2026-02-25 09:15:00 | 934.85 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2026-03-04 10:30:00 | 979.45 | 2026-03-05 14:15:00 | 1077.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 993.50 | 2026-03-13 12:15:00 | 943.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 993.50 | 2026-03-13 15:15:00 | 969.95 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2026-03-24 10:15:00 | 998.15 | 2026-03-24 15:15:00 | 1000.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-03-24 13:15:00 | 998.55 | 2026-03-24 15:15:00 | 1000.85 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-04-01 10:15:00 | 977.55 | 2026-04-07 09:15:00 | 928.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 977.55 | 2026-04-07 12:15:00 | 943.65 | STOP_HIT | 0.50 | 3.47% |
| BUY | retest2 | 2026-04-13 13:30:00 | 995.25 | 2026-04-16 09:15:00 | 969.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-04-13 14:30:00 | 994.55 | 2026-04-16 09:15:00 | 969.50 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1000.70 | 2026-04-16 09:15:00 | 969.50 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-17 12:00:00 | 973.75 | 2026-04-22 10:15:00 | 984.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-04-17 12:45:00 | 970.25 | 2026-04-22 10:15:00 | 984.65 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-21 11:30:00 | 962.15 | 2026-04-22 10:15:00 | 984.65 | STOP_HIT | 1.00 | -2.34% |
