# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 951.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 18 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 34 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 28
- **Target hits / Stop hits / Partials:** 1 / 37 / 8
- **Avg / median % per leg:** 0.94% / -0.67%
- **Sum % (uncompounded):** 43.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 1 | 19 | 1 | 0.18% | 3.7% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 1.64% | 8.2% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 1 | 15 | 0 | -0.28% | -4.5% |
| SELL (all) | 25 | 14 | 56.0% | 0 | 18 | 7 | 1.59% | 39.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 14 | 56.0% | 0 | 18 | 7 | 1.59% | 39.6% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 1.64% | 8.2% |
| retest2 (combined) | 41 | 16 | 39.0% | 1 | 33 | 7 | 0.86% | 35.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 676.05 | 670.92 | 670.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 677.70 | 672.93 | 671.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 671.45 | 672.63 | 671.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 671.45 | 672.63 | 671.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 671.45 | 672.63 | 671.70 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 11:15:00 | 663.10 | 669.76 | 670.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 662.15 | 668.24 | 669.73 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 684.55 | 670.51 | 670.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 700.20 | 687.37 | 680.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 11:15:00 | 697.60 | 699.23 | 692.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:45:00 | 695.60 | 699.23 | 692.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 695.00 | 698.77 | 694.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 684.70 | 698.77 | 694.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 691.00 | 697.21 | 694.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 691.00 | 697.21 | 694.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 687.30 | 695.23 | 693.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 687.30 | 695.23 | 693.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 685.40 | 691.95 | 692.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 677.05 | 686.58 | 689.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 15:15:00 | 681.95 | 681.26 | 685.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 685.25 | 682.05 | 685.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 685.25 | 682.05 | 685.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 685.25 | 682.05 | 685.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 677.50 | 681.14 | 684.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:30:00 | 670.75 | 679.75 | 683.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 672.10 | 679.08 | 682.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 672.25 | 677.72 | 682.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 716.45 | 683.66 | 683.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 716.45 | 683.66 | 683.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 735.40 | 700.44 | 691.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 718.15 | 718.85 | 711.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:30:00 | 718.25 | 718.85 | 711.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 712.00 | 716.62 | 711.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 712.00 | 716.62 | 711.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 706.40 | 714.57 | 711.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 706.40 | 714.57 | 711.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 707.95 | 713.25 | 710.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 705.35 | 713.25 | 710.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 711.35 | 711.44 | 710.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 711.15 | 711.44 | 710.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 708.00 | 710.76 | 710.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 705.00 | 710.76 | 710.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 700.35 | 708.67 | 709.49 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 717.95 | 710.05 | 709.57 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 708.10 | 711.25 | 711.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 707.05 | 710.41 | 711.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 713.95 | 711.02 | 711.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 713.95 | 711.02 | 711.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 713.95 | 711.02 | 711.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 713.95 | 711.02 | 711.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 716.95 | 712.21 | 711.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 729.45 | 715.65 | 713.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 15:15:00 | 720.25 | 720.98 | 717.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:15:00 | 732.70 | 720.98 | 717.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 731.45 | 728.98 | 724.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 723.75 | 728.98 | 724.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 766.40 | 764.11 | 758.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 760.40 | 764.11 | 758.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 10:15:00 | 769.34 | 765.44 | 759.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 770.40 | 772.02 | 765.88 | SL hit (close<ema200) qty=0.50 sl=772.02 alert=retest1 |

### Cycle 10 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 759.85 | 765.69 | 766.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 758.50 | 764.25 | 765.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 757.20 | 754.71 | 758.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 757.20 | 754.71 | 758.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 759.40 | 755.65 | 758.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 759.40 | 755.65 | 758.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 761.65 | 756.85 | 758.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 763.30 | 756.85 | 758.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 766.00 | 759.74 | 759.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 10:15:00 | 767.35 | 762.43 | 760.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 760.00 | 762.28 | 761.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 760.00 | 762.28 | 761.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 760.00 | 762.28 | 761.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:30:00 | 757.80 | 762.28 | 761.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 762.75 | 762.37 | 761.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:30:00 | 756.95 | 762.37 | 761.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 762.40 | 762.38 | 761.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 761.65 | 762.38 | 761.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 761.50 | 762.20 | 761.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 764.45 | 762.20 | 761.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 758.65 | 760.75 | 760.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 758.65 | 760.75 | 760.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 757.25 | 760.05 | 760.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 753.00 | 751.54 | 754.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 753.00 | 751.54 | 754.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 749.55 | 751.14 | 754.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 748.40 | 750.38 | 753.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 755.75 | 751.45 | 753.50 | SL hit (close>static) qty=1.00 sl=754.85 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 758.85 | 754.62 | 754.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 778.45 | 765.58 | 760.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 789.40 | 791.00 | 783.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 789.40 | 791.00 | 783.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 784.80 | 790.10 | 784.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 784.80 | 790.10 | 784.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 783.00 | 788.68 | 784.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 793.10 | 788.68 | 784.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 785.90 | 787.71 | 785.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 777.10 | 786.68 | 785.99 | SL hit (close<static) qty=1.00 sl=779.95 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 776.40 | 784.63 | 785.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 773.05 | 778.69 | 781.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 782.35 | 777.34 | 780.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 782.35 | 777.34 | 780.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 782.35 | 777.34 | 780.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:45:00 | 783.75 | 777.34 | 780.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 785.60 | 778.99 | 780.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 785.60 | 778.99 | 780.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 792.70 | 783.80 | 782.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 799.30 | 789.87 | 786.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 804.00 | 805.03 | 800.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 804.00 | 805.03 | 800.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 804.00 | 805.03 | 800.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 801.85 | 805.03 | 800.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 799.80 | 805.36 | 802.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 799.80 | 805.36 | 802.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 795.35 | 803.35 | 801.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 792.40 | 803.35 | 801.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 789.90 | 799.02 | 799.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 786.90 | 791.31 | 795.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 774.10 | 767.57 | 773.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 774.10 | 767.57 | 773.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 774.10 | 767.57 | 773.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 774.30 | 767.57 | 773.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 774.45 | 768.95 | 773.72 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 780.45 | 776.29 | 776.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 785.10 | 778.05 | 776.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 798.80 | 799.26 | 793.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:30:00 | 797.00 | 799.26 | 793.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 808.70 | 805.41 | 800.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 810.25 | 805.41 | 800.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 798.15 | 803.38 | 801.38 | SL hit (close<static) qty=1.00 sl=799.65 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 798.10 | 801.03 | 801.18 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 812.25 | 803.27 | 802.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 817.85 | 810.96 | 806.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 13:15:00 | 836.45 | 840.14 | 832.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 14:00:00 | 836.45 | 840.14 | 832.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 844.70 | 841.05 | 833.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 836.30 | 841.05 | 833.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 844.20 | 845.78 | 841.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:45:00 | 842.60 | 845.78 | 841.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 847.00 | 846.02 | 842.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 840.90 | 846.02 | 842.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 852.85 | 847.39 | 843.00 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 840.00 | 843.31 | 843.57 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 847.00 | 843.75 | 843.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 850.60 | 845.12 | 844.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 855.40 | 857.91 | 853.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 855.40 | 857.91 | 853.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 855.40 | 857.91 | 853.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 853.60 | 857.91 | 853.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 856.15 | 857.56 | 854.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 851.70 | 857.56 | 854.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 855.45 | 859.42 | 856.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 854.35 | 859.42 | 856.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 861.90 | 859.91 | 857.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 862.15 | 859.91 | 857.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:30:00 | 862.60 | 860.73 | 858.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 849.90 | 860.07 | 859.22 | SL hit (close<static) qty=1.00 sl=855.10 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 852.25 | 858.50 | 858.59 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 09:15:00 | 884.20 | 863.08 | 860.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 907.00 | 897.06 | 886.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 937.80 | 938.21 | 927.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 937.80 | 938.21 | 927.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 942.00 | 940.66 | 936.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:30:00 | 936.30 | 940.66 | 936.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 947.40 | 953.46 | 950.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 947.50 | 953.46 | 950.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 948.10 | 952.39 | 950.11 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 939.60 | 948.81 | 948.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 930.00 | 937.09 | 941.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 15:15:00 | 917.65 | 916.80 | 923.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 920.50 | 916.80 | 923.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 911.30 | 915.70 | 922.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:30:00 | 907.10 | 911.58 | 916.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 922.40 | 917.70 | 917.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 922.40 | 917.70 | 917.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 923.95 | 920.49 | 918.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 948.80 | 948.82 | 941.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:30:00 | 949.00 | 948.82 | 941.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 943.80 | 948.12 | 944.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 943.80 | 948.12 | 944.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 948.15 | 948.13 | 945.00 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 934.80 | 943.30 | 943.65 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 946.90 | 942.55 | 942.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 954.85 | 945.01 | 943.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 963.10 | 964.58 | 960.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 963.10 | 964.58 | 960.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 971.50 | 968.08 | 963.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:00:00 | 973.30 | 969.90 | 965.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 973.30 | 970.58 | 966.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 974.20 | 972.16 | 968.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 957.25 | 965.96 | 966.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 957.25 | 965.96 | 966.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 953.35 | 963.44 | 965.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 963.40 | 962.63 | 964.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 963.40 | 962.63 | 964.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 963.25 | 962.76 | 964.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 962.75 | 962.76 | 964.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 958.70 | 961.95 | 964.07 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 970.05 | 965.05 | 964.86 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 955.40 | 963.28 | 964.10 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 15:15:00 | 970.05 | 961.67 | 960.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 10:15:00 | 973.40 | 965.63 | 962.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 967.55 | 969.05 | 965.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 967.55 | 969.05 | 965.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 968.90 | 969.02 | 966.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 963.35 | 969.02 | 966.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 956.00 | 966.41 | 965.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 956.00 | 966.41 | 965.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 956.35 | 964.40 | 964.35 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 959.65 | 963.45 | 963.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 951.90 | 959.73 | 961.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 945.50 | 938.98 | 945.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 945.50 | 938.98 | 945.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 945.50 | 938.98 | 945.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 947.10 | 938.98 | 945.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 938.10 | 938.80 | 944.91 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 965.60 | 948.01 | 947.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 971.40 | 952.69 | 950.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 13:15:00 | 959.20 | 961.79 | 956.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 959.20 | 961.79 | 956.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 977.00 | 983.37 | 976.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 977.00 | 983.37 | 976.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 975.95 | 981.89 | 975.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 1007.45 | 980.29 | 976.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1066.70 | 1086.11 | 1088.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 1066.70 | 1086.11 | 1088.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 1058.35 | 1080.56 | 1085.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 1046.35 | 1045.35 | 1058.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 1051.80 | 1045.35 | 1058.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1050.90 | 1046.66 | 1054.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 1053.10 | 1046.66 | 1054.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1053.95 | 1048.12 | 1054.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:15:00 | 1051.15 | 1048.12 | 1054.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1051.15 | 1048.73 | 1054.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1060.80 | 1048.73 | 1054.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1061.00 | 1051.18 | 1054.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1063.00 | 1051.18 | 1054.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1058.25 | 1052.59 | 1055.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1057.00 | 1052.59 | 1055.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1057.70 | 1055.47 | 1055.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:15:00 | 1060.45 | 1055.47 | 1055.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1060.45 | 1056.46 | 1056.35 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 1053.15 | 1056.32 | 1056.34 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 1058.05 | 1056.38 | 1056.32 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 14:15:00 | 1053.55 | 1055.82 | 1056.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 1053.05 | 1054.86 | 1055.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 1032.30 | 1031.60 | 1039.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:45:00 | 1031.30 | 1031.60 | 1039.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1037.40 | 1032.76 | 1039.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 1037.40 | 1032.76 | 1039.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1039.90 | 1034.71 | 1038.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1039.90 | 1034.71 | 1038.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1031.90 | 1034.15 | 1038.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 1038.70 | 1034.15 | 1038.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 999.90 | 1019.04 | 1027.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 990.00 | 1008.44 | 1013.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 994.10 | 1006.02 | 1012.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 992.00 | 1003.21 | 1010.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:15:00 | 940.50 | 955.54 | 966.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:15:00 | 944.39 | 955.54 | 966.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:15:00 | 942.40 | 955.54 | 966.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 944.40 | 940.48 | 949.80 | SL hit (close>ema200) qty=0.50 sl=940.48 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 937.00 | 933.50 | 933.33 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 922.70 | 931.34 | 932.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 917.50 | 928.57 | 931.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 929.40 | 925.50 | 927.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 929.40 | 925.50 | 927.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 929.40 | 925.50 | 927.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 928.80 | 925.50 | 927.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 930.40 | 926.48 | 928.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 923.40 | 926.26 | 927.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:15:00 | 925.10 | 924.48 | 926.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 925.00 | 924.54 | 926.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 932.20 | 925.82 | 925.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 932.20 | 925.82 | 925.68 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 919.60 | 926.00 | 926.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 910.30 | 919.68 | 922.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 919.30 | 916.64 | 920.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 919.30 | 916.64 | 920.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 920.00 | 917.31 | 920.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 918.35 | 917.31 | 920.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:45:00 | 917.00 | 917.79 | 919.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 914.15 | 917.10 | 919.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 872.43 | 885.99 | 891.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 871.15 | 882.46 | 889.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 868.44 | 882.46 | 889.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 864.90 | 864.10 | 872.28 | SL hit (close>ema200) qty=0.50 sl=864.10 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 873.95 | 862.11 | 861.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 12:15:00 | 878.35 | 871.91 | 869.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 15:15:00 | 872.90 | 873.15 | 870.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:15:00 | 877.10 | 873.15 | 870.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 11:00:00 | 876.60 | 873.86 | 871.20 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 871.30 | 873.50 | 871.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 871.30 | 873.50 | 871.51 | SL hit (close<ema400) qty=1.00 sl=871.51 alert=retest1 |

### Cycle 44 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 859.15 | 868.96 | 869.86 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 882.55 | 871.16 | 869.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 890.70 | 880.00 | 874.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 901.50 | 904.04 | 894.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:45:00 | 900.75 | 904.04 | 894.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 899.85 | 903.13 | 900.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 899.85 | 903.13 | 900.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 904.00 | 903.31 | 900.92 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 890.25 | 901.17 | 901.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 889.50 | 897.57 | 899.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 884.05 | 881.77 | 886.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 884.05 | 881.77 | 886.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 886.30 | 883.65 | 886.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 886.30 | 883.65 | 886.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 883.35 | 883.59 | 886.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 886.40 | 883.59 | 886.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 881.60 | 883.38 | 885.58 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 894.20 | 886.77 | 886.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 900.10 | 889.44 | 887.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 912.25 | 913.80 | 906.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 912.25 | 913.80 | 906.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 929.10 | 938.01 | 931.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 929.10 | 938.01 | 931.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 932.40 | 936.89 | 931.96 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 915.85 | 927.01 | 928.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 911.25 | 920.54 | 924.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 895.35 | 893.76 | 901.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:30:00 | 899.20 | 893.76 | 901.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 899.80 | 895.73 | 901.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 900.50 | 895.73 | 901.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 907.90 | 898.16 | 901.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 907.90 | 898.16 | 901.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 910.00 | 900.53 | 902.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 914.00 | 900.53 | 902.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 913.05 | 905.35 | 904.36 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 901.15 | 904.68 | 905.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 896.50 | 902.29 | 903.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 854.30 | 851.40 | 863.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 854.65 | 851.40 | 863.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 852.00 | 846.92 | 854.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 853.50 | 846.92 | 854.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 856.50 | 848.84 | 854.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 857.55 | 848.84 | 854.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 853.20 | 849.71 | 854.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:30:00 | 847.85 | 850.59 | 854.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 850.00 | 846.65 | 846.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 850.00 | 846.65 | 846.63 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 841.50 | 845.62 | 846.16 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 853.30 | 845.64 | 845.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 854.15 | 847.34 | 846.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 848.50 | 849.51 | 848.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 848.50 | 849.51 | 848.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 848.50 | 849.51 | 848.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 857.25 | 851.06 | 848.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 836.15 | 847.88 | 848.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 836.15 | 847.88 | 848.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 831.30 | 840.36 | 844.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 838.40 | 831.55 | 837.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 838.40 | 831.55 | 837.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 838.40 | 831.55 | 837.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 838.40 | 831.55 | 837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 840.00 | 833.24 | 837.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 861.15 | 833.24 | 837.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 862.35 | 842.69 | 841.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 866.50 | 847.45 | 843.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 859.90 | 861.14 | 855.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 859.90 | 861.14 | 855.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 854.70 | 859.32 | 855.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 854.70 | 859.32 | 855.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 852.80 | 858.02 | 855.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 852.80 | 858.02 | 855.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 853.20 | 857.05 | 855.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 856.40 | 856.33 | 855.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 856.75 | 856.40 | 855.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 848.30 | 854.88 | 854.79 | SL hit (close<static) qty=1.00 sl=851.85 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 847.95 | 853.49 | 854.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 865.35 | 855.69 | 854.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 885.65 | 861.68 | 857.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 13:15:00 | 927.00 | 927.18 | 913.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 927.00 | 927.18 | 913.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 917.00 | 925.56 | 915.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 919.80 | 925.56 | 915.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 922.95 | 925.04 | 916.56 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 904.00 | 915.97 | 915.98 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 914.80 | 912.02 | 911.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 920.85 | 913.78 | 912.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 914.00 | 914.75 | 913.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 914.00 | 914.75 | 913.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 914.00 | 914.75 | 913.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 916.75 | 914.75 | 913.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 905.80 | 912.96 | 912.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 905.80 | 912.96 | 912.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 905.05 | 911.38 | 912.14 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 920.50 | 913.99 | 913.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 922.00 | 916.50 | 914.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 916.25 | 917.28 | 915.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:15:00 | 923.40 | 917.28 | 915.30 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 917.20 | 920.92 | 918.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 917.20 | 920.92 | 918.06 | SL hit (close<ema400) qty=1.00 sl=918.06 alert=retest1 |

### Cycle 62 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 911.45 | 916.20 | 916.78 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 923.50 | 918.01 | 917.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 924.30 | 919.27 | 918.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 949.50 | 952.72 | 943.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 951.05 | 952.72 | 943.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 941.10 | 950.22 | 944.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 941.10 | 950.22 | 944.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 939.85 | 948.15 | 944.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 930.35 | 948.15 | 944.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 928.30 | 940.32 | 941.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 926.35 | 937.53 | 939.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 920.15 | 913.98 | 919.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 920.15 | 913.98 | 919.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 920.15 | 913.98 | 919.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 920.15 | 913.98 | 919.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 920.50 | 915.28 | 919.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 927.45 | 915.28 | 919.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 917.00 | 915.63 | 919.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 923.55 | 915.63 | 919.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 909.65 | 914.43 | 918.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:45:00 | 907.40 | 912.33 | 917.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 862.03 | 895.64 | 906.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 882.60 | 881.78 | 891.29 | SL hit (close>ema200) qty=0.50 sl=881.78 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 843.80 | 834.38 | 833.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 848.75 | 838.89 | 835.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 820.10 | 836.35 | 835.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 820.10 | 836.35 | 835.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 820.10 | 836.35 | 835.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 812.90 | 836.35 | 835.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 816.75 | 832.43 | 833.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 792.30 | 817.71 | 823.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 800.00 | 798.81 | 810.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:45:00 | 799.55 | 798.81 | 810.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 803.15 | 799.38 | 808.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 798.20 | 799.89 | 807.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 829.90 | 811.81 | 810.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 829.90 | 811.81 | 810.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 832.00 | 825.64 | 819.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 824.35 | 825.38 | 819.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 824.35 | 825.38 | 819.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 824.35 | 825.38 | 819.83 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 809.10 | 816.73 | 817.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 794.95 | 812.37 | 815.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 810.70 | 803.75 | 807.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 810.70 | 803.75 | 807.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 810.70 | 803.75 | 807.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 803.60 | 803.75 | 807.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 803.70 | 795.41 | 795.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 803.70 | 795.41 | 795.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 808.35 | 798.00 | 796.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 848.50 | 851.15 | 844.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:30:00 | 846.50 | 851.15 | 844.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 841.70 | 848.76 | 844.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 848.30 | 849.79 | 845.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 10:15:00 | 933.13 | 919.87 | 906.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 933.65 | 942.25 | 942.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 919.00 | 937.60 | 940.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 955.35 | 930.06 | 933.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 955.35 | 930.06 | 933.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 955.35 | 930.06 | 933.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 955.35 | 930.06 | 933.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 952.40 | 934.53 | 934.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 953.00 | 934.53 | 934.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 951.20 | 937.87 | 936.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 962.50 | 953.03 | 950.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 952.90 | 954.06 | 951.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 952.90 | 954.06 | 951.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 952.30 | 953.71 | 951.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 951.30 | 953.71 | 951.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 951.00 | 953.16 | 951.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 948.50 | 953.16 | 951.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 951.20 | 952.77 | 951.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 951.30 | 952.77 | 951.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 951.30 | 952.48 | 951.32 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 11:30:00 | 670.75 | 2025-05-22 09:15:00 | 716.45 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2025-05-21 12:30:00 | 672.10 | 2025-05-22 09:15:00 | 716.45 | STOP_HIT | 1.00 | -6.60% |
| SELL | retest2 | 2025-05-21 14:00:00 | 672.25 | 2025-05-22 09:15:00 | 716.45 | STOP_HIT | 1.00 | -6.57% |
| BUY | retest1 | 2025-06-03 09:15:00 | 732.70 | 2025-06-10 10:15:00 | 769.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-03 09:15:00 | 732.70 | 2025-06-11 09:15:00 | 770.40 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2025-06-12 11:15:00 | 772.65 | 2025-06-12 13:15:00 | 762.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-18 09:15:00 | 764.45 | 2025-06-18 12:15:00 | 758.65 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-20 14:15:00 | 748.40 | 2025-06-20 14:15:00 | 755.75 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-30 09:15:00 | 793.10 | 2025-07-01 10:15:00 | 777.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-06-30 13:00:00 | 785.90 | 2025-07-01 10:15:00 | 777.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-18 11:15:00 | 810.25 | 2025-07-18 14:15:00 | 798.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-21 13:15:00 | 808.75 | 2025-07-22 10:15:00 | 798.10 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-05 11:15:00 | 862.15 | 2025-08-06 10:15:00 | 849.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-05 12:30:00 | 862.60 | 2025-08-06 10:15:00 | 849.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-29 09:30:00 | 907.10 | 2025-09-01 11:15:00 | 922.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-15 13:00:00 | 973.30 | 2025-09-17 10:15:00 | 957.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-15 14:00:00 | 973.30 | 2025-09-17 10:15:00 | 957.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-16 10:45:00 | 974.20 | 2025-09-17 10:15:00 | 957.25 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-06 09:15:00 | 1007.45 | 2025-10-23 11:15:00 | 1066.70 | STOP_HIT | 1.00 | 5.88% |
| SELL | retest2 | 2025-11-10 10:45:00 | 990.00 | 2025-11-14 09:15:00 | 940.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 11:30:00 | 994.10 | 2025-11-14 09:15:00 | 944.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 13:00:00 | 992.00 | 2025-11-14 09:15:00 | 942.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 10:45:00 | 990.00 | 2025-11-17 11:15:00 | 944.40 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-11-10 11:30:00 | 994.10 | 2025-11-17 11:15:00 | 944.40 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 13:00:00 | 992.00 | 2025-11-17 11:15:00 | 944.40 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2025-11-24 15:00:00 | 923.40 | 2025-11-26 11:15:00 | 932.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-25 10:15:00 | 925.10 | 2025-11-26 11:15:00 | 932.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-25 11:45:00 | 925.00 | 2025-11-26 11:15:00 | 932.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-01 09:15:00 | 918.35 | 2025-12-08 09:15:00 | 872.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:45:00 | 917.00 | 2025-12-08 10:15:00 | 871.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 914.15 | 2025-12-08 10:15:00 | 868.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 918.35 | 2025-12-09 13:15:00 | 864.90 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2025-12-01 09:45:00 | 917.00 | 2025-12-09 13:15:00 | 864.90 | STOP_HIT | 0.50 | 5.68% |
| SELL | retest2 | 2025-12-01 10:45:00 | 914.15 | 2025-12-09 13:15:00 | 864.90 | STOP_HIT | 0.50 | 5.39% |
| BUY | retest1 | 2025-12-17 09:15:00 | 877.10 | 2025-12-17 12:15:00 | 871.30 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2025-12-17 11:00:00 | 876.60 | 2025-12-17 12:15:00 | 871.30 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-23 13:30:00 | 847.85 | 2026-01-28 15:15:00 | 850.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-01 11:00:00 | 857.25 | 2026-02-01 13:15:00 | 836.15 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-02-05 13:30:00 | 856.40 | 2026-02-06 09:15:00 | 848.30 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-05 14:30:00 | 856.75 | 2026-02-06 09:15:00 | 848.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest1 | 2026-02-23 09:15:00 | 923.40 | 2026-02-23 12:15:00 | 917.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-03-06 11:45:00 | 907.40 | 2026-03-09 09:15:00 | 862.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 11:45:00 | 907.40 | 2026-03-10 10:15:00 | 882.60 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2026-03-24 10:30:00 | 798.20 | 2026-03-25 09:15:00 | 829.90 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-04-01 10:15:00 | 803.60 | 2026-04-06 13:15:00 | 803.70 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-04-13 10:45:00 | 848.30 | 2026-04-22 10:15:00 | 933.13 | TARGET_HIT | 1.00 | 10.00% |
