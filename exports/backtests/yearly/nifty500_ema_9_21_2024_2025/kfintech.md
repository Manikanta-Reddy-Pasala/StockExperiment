# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 917.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 102 |
| ALERT2 | 99 |
| ALERT2_SKIP | 50 |
| ALERT3 | 263 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 121 |
| PARTIAL | 11 |
| TARGET_HIT | 12 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 100
- **Target hits / Stop hits / Partials:** 12 / 112 / 11
- **Avg / median % per leg:** 0.12% / -1.13%
- **Sum % (uncompounded):** 15.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 9 | 22.5% | 9 | 31 | 0 | 0.71% | 28.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.60% | -10.8% |
| BUY @ 3rd Alert (retest2) | 37 | 9 | 24.3% | 9 | 28 | 0 | 1.06% | 39.2% |
| SELL (all) | 95 | 26 | 27.4% | 3 | 81 | 11 | -0.13% | -12.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 95 | 26 | 27.4% | 3 | 81 | 11 | -0.13% | -12.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.60% | -10.8% |
| retest2 (combined) | 132 | 35 | 26.5% | 12 | 109 | 11 | 0.20% | 26.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 758.45 | 754.73 | 754.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 774.00 | 759.21 | 756.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 10:15:00 | 762.10 | 762.33 | 759.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:00:00 | 762.10 | 762.33 | 759.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 759.25 | 761.71 | 759.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:00:00 | 759.25 | 761.71 | 759.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 759.25 | 761.22 | 759.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:30:00 | 759.15 | 761.22 | 759.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 757.85 | 760.55 | 759.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:30:00 | 756.35 | 760.55 | 759.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 759.55 | 760.35 | 759.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 759.55 | 760.35 | 759.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 761.00 | 760.48 | 759.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 751.00 | 758.58 | 758.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 745.20 | 755.91 | 757.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 12:15:00 | 740.80 | 751.13 | 754.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 13:15:00 | 744.95 | 743.15 | 747.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 14:00:00 | 744.95 | 743.15 | 747.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 753.95 | 745.31 | 748.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 753.95 | 745.31 | 748.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 754.45 | 747.14 | 748.67 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 764.50 | 750.61 | 750.11 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 741.50 | 749.17 | 749.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 733.10 | 744.24 | 746.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 743.25 | 742.76 | 745.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 12:00:00 | 743.25 | 742.76 | 745.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 748.80 | 743.97 | 745.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 750.15 | 743.97 | 745.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 752.70 | 745.71 | 746.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 752.70 | 745.71 | 746.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 14:15:00 | 753.60 | 747.29 | 747.19 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 745.20 | 747.16 | 747.21 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 753.25 | 748.38 | 747.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 755.00 | 750.42 | 748.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 11:15:00 | 750.70 | 752.59 | 750.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 750.70 | 752.59 | 750.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 750.70 | 752.59 | 750.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 750.70 | 752.59 | 750.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 748.70 | 751.81 | 750.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:15:00 | 748.45 | 751.81 | 750.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 746.15 | 750.68 | 750.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:30:00 | 749.60 | 750.68 | 750.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 745.85 | 749.71 | 749.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:30:00 | 746.25 | 749.71 | 749.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 744.50 | 748.67 | 749.22 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 753.80 | 750.21 | 749.83 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 744.25 | 750.18 | 750.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 737.90 | 747.73 | 749.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 746.85 | 746.36 | 748.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:45:00 | 745.35 | 746.36 | 748.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 744.00 | 745.88 | 747.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 748.35 | 745.88 | 747.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 704.85 | 697.19 | 706.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 704.85 | 697.19 | 706.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 712.85 | 700.33 | 707.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:45:00 | 713.50 | 700.33 | 707.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 714.75 | 703.21 | 707.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 714.75 | 703.21 | 707.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 719.75 | 710.87 | 710.57 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 693.90 | 708.82 | 709.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 648.40 | 696.73 | 704.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 683.30 | 676.01 | 685.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 684.70 | 676.01 | 685.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 693.55 | 680.20 | 685.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 693.50 | 680.20 | 685.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 693.00 | 682.76 | 686.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 694.45 | 682.76 | 686.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 693.75 | 684.96 | 687.11 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 691.55 | 688.58 | 688.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 715.00 | 695.29 | 691.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 713.60 | 714.30 | 707.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:15:00 | 714.60 | 714.30 | 707.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 713.25 | 713.87 | 708.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 718.50 | 713.87 | 708.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 716.90 | 724.69 | 725.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 716.90 | 724.69 | 725.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 11:15:00 | 711.60 | 717.88 | 720.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 685.95 | 685.93 | 694.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 685.95 | 685.93 | 694.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 700.20 | 689.55 | 694.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 700.20 | 689.55 | 694.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 705.50 | 692.74 | 695.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 705.50 | 692.74 | 695.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 699.80 | 696.45 | 696.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 15:15:00 | 689.95 | 695.03 | 696.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 689.70 | 693.82 | 695.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:30:00 | 689.20 | 691.77 | 693.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 15:15:00 | 687.95 | 690.39 | 692.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 695.35 | 690.99 | 692.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 695.70 | 693.88 | 693.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 695.70 | 693.88 | 693.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 697.55 | 694.61 | 694.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 696.10 | 696.64 | 695.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 696.10 | 696.64 | 695.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 696.10 | 696.64 | 695.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 696.10 | 696.64 | 695.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 694.25 | 696.16 | 695.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 693.85 | 696.16 | 695.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 692.35 | 695.40 | 695.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 692.35 | 695.40 | 695.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 706.00 | 697.52 | 696.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 713.75 | 700.00 | 698.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-10 09:15:00 | 785.13 | 765.19 | 759.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 759.90 | 761.22 | 761.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 753.40 | 759.30 | 760.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 758.40 | 753.45 | 755.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 758.40 | 753.45 | 755.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 758.40 | 753.45 | 755.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 756.75 | 753.45 | 755.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 759.90 | 754.74 | 756.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:30:00 | 763.25 | 754.74 | 756.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 754.20 | 754.63 | 755.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 14:30:00 | 753.55 | 755.23 | 755.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 09:45:00 | 754.05 | 754.43 | 755.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 10:45:00 | 754.10 | 754.05 | 755.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 14:00:00 | 753.15 | 754.27 | 755.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 754.10 | 754.23 | 754.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 752.00 | 754.43 | 754.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 768.90 | 756.89 | 755.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 10:15:00 | 768.90 | 756.89 | 755.98 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 742.65 | 758.23 | 758.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 728.00 | 746.11 | 752.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 748.80 | 737.67 | 742.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 15:15:00 | 748.80 | 737.67 | 742.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 748.80 | 737.67 | 742.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 726.15 | 734.93 | 741.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 726.80 | 734.90 | 740.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 725.70 | 730.72 | 737.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:45:00 | 727.30 | 730.38 | 736.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 740.25 | 732.13 | 736.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 739.75 | 732.13 | 736.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 741.80 | 734.06 | 736.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 741.80 | 734.06 | 736.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 732.50 | 733.75 | 736.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 731.80 | 733.75 | 736.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 742.25 | 737.62 | 737.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 742.25 | 737.62 | 737.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 768.30 | 745.58 | 741.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 876.65 | 877.62 | 860.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 09:45:00 | 880.05 | 877.62 | 860.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 860.60 | 872.59 | 863.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:45:00 | 859.95 | 872.59 | 863.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 857.60 | 869.59 | 863.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 857.00 | 869.59 | 863.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 857.95 | 867.26 | 862.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 819.00 | 867.26 | 862.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 822.20 | 854.80 | 857.54 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 860.45 | 847.67 | 846.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 884.10 | 854.96 | 850.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 10:15:00 | 1010.60 | 1012.84 | 961.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:30:00 | 1023.85 | 1013.39 | 966.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 15:00:00 | 1025.95 | 1014.07 | 978.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:45:00 | 1029.50 | 1022.87 | 988.65 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1021.95 | 1025.31 | 1005.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 998.80 | 1025.31 | 1005.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 989.50 | 1018.15 | 1004.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 989.50 | 1018.15 | 1004.28 | SL hit (close<ema400) qty=1.00 sl=1004.28 alert=retest1 |

### Cycle 22 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 997.15 | 1008.25 | 1008.93 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 1012.55 | 1008.47 | 1008.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 1060.80 | 1018.94 | 1012.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 1023.05 | 1026.60 | 1019.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 13:00:00 | 1023.05 | 1026.60 | 1019.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1019.00 | 1025.08 | 1019.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 1019.00 | 1025.08 | 1019.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1015.20 | 1023.10 | 1019.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 1015.20 | 1023.10 | 1019.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1020.10 | 1022.50 | 1019.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1006.50 | 1022.50 | 1019.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1002.75 | 1018.55 | 1017.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1002.75 | 1018.55 | 1017.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 998.25 | 1014.49 | 1016.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 11:15:00 | 987.85 | 1009.16 | 1013.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1000.40 | 998.72 | 1005.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 1000.40 | 998.72 | 1005.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1000.40 | 998.72 | 1005.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 1003.25 | 998.72 | 1005.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1001.00 | 999.17 | 1005.32 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 1084.80 | 1021.29 | 1013.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 13:15:00 | 1152.75 | 1069.73 | 1040.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 10:15:00 | 1092.95 | 1097.29 | 1065.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 11:15:00 | 1091.00 | 1097.29 | 1065.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1073.70 | 1088.06 | 1071.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1073.70 | 1088.06 | 1071.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1076.00 | 1085.65 | 1071.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1065.65 | 1085.65 | 1071.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1054.75 | 1079.47 | 1069.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 1054.75 | 1079.47 | 1069.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1043.10 | 1072.20 | 1067.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1043.10 | 1072.20 | 1067.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1039.80 | 1060.40 | 1062.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1035.00 | 1055.32 | 1060.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 1045.65 | 1039.90 | 1045.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 1045.65 | 1039.90 | 1045.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1045.65 | 1039.90 | 1045.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 1032.80 | 1039.46 | 1044.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 15:15:00 | 981.16 | 997.36 | 1004.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 1000.00 | 995.82 | 1002.15 | SL hit (close>ema200) qty=0.50 sl=995.82 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1025.80 | 997.41 | 994.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 1036.35 | 1022.97 | 1014.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 1019.60 | 1022.30 | 1015.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:30:00 | 1022.00 | 1022.30 | 1015.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1011.90 | 1032.12 | 1027.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 1011.90 | 1032.12 | 1027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1011.40 | 1027.98 | 1026.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 1005.75 | 1027.98 | 1026.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 997.80 | 1021.94 | 1023.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 995.55 | 1016.66 | 1021.13 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 1064.25 | 1021.36 | 1020.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 1079.05 | 1032.90 | 1025.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 1091.95 | 1097.74 | 1080.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 1091.95 | 1097.74 | 1080.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1038.05 | 1082.69 | 1081.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 1046.45 | 1082.69 | 1081.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1040.70 | 1074.29 | 1077.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 1036.60 | 1066.75 | 1074.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1082.00 | 1040.95 | 1044.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1082.00 | 1040.95 | 1044.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1082.00 | 1040.95 | 1044.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 1085.95 | 1040.95 | 1044.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 1079.95 | 1048.75 | 1048.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 1111.00 | 1061.20 | 1053.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 1098.10 | 1106.00 | 1086.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 12:00:00 | 1098.10 | 1106.00 | 1086.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 1091.50 | 1103.10 | 1087.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 1091.50 | 1103.10 | 1087.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1083.90 | 1099.26 | 1086.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1083.90 | 1099.26 | 1086.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1090.00 | 1097.41 | 1087.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 1072.00 | 1097.41 | 1087.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1072.00 | 1092.33 | 1085.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1039.65 | 1092.33 | 1085.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1045.70 | 1083.00 | 1082.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1038.00 | 1083.00 | 1082.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 1048.35 | 1076.07 | 1079.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 1026.40 | 1054.63 | 1067.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 13:15:00 | 1003.20 | 1000.34 | 1016.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:45:00 | 1002.00 | 1000.34 | 1016.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1056.50 | 1013.48 | 1018.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 1050.70 | 1013.48 | 1018.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1036.00 | 1017.99 | 1020.21 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1041.20 | 1025.54 | 1023.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 1048.50 | 1036.17 | 1031.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 15:15:00 | 1042.95 | 1043.45 | 1038.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:15:00 | 1033.65 | 1043.45 | 1038.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1029.65 | 1040.69 | 1037.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 1029.65 | 1040.69 | 1037.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1031.55 | 1038.87 | 1036.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 1029.25 | 1038.87 | 1036.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1081.00 | 1046.69 | 1040.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1105.50 | 1061.33 | 1048.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 1097.80 | 1073.10 | 1056.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:45:00 | 1097.95 | 1078.39 | 1060.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:00:00 | 1103.00 | 1086.38 | 1068.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1066.50 | 1095.60 | 1090.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 1066.50 | 1095.60 | 1090.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1071.95 | 1090.87 | 1089.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1042.20 | 1090.87 | 1089.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1057.30 | 1084.16 | 1086.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 1057.30 | 1084.16 | 1086.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 1029.70 | 1045.47 | 1059.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 1007.80 | 997.92 | 1017.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 1007.80 | 997.92 | 1017.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 958.00 | 973.53 | 989.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 955.25 | 973.53 | 989.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 953.70 | 952.04 | 967.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:45:00 | 952.15 | 958.46 | 964.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:00:00 | 953.70 | 955.14 | 961.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 964.25 | 957.60 | 961.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 964.25 | 957.60 | 961.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 965.90 | 959.26 | 962.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:45:00 | 963.75 | 959.26 | 962.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 975.65 | 962.54 | 963.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 981.00 | 962.54 | 963.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 974.00 | 964.83 | 964.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 974.00 | 964.83 | 964.43 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 951.00 | 965.10 | 965.69 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 987.05 | 967.95 | 966.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1015.70 | 990.55 | 979.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 994.30 | 995.18 | 984.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 11:00:00 | 994.30 | 995.18 | 984.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1029.00 | 1036.60 | 1031.01 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 1021.75 | 1026.99 | 1027.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 1020.00 | 1025.59 | 1026.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 12:15:00 | 1017.70 | 1016.60 | 1021.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 12:15:00 | 1017.70 | 1016.60 | 1021.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1017.70 | 1016.60 | 1021.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 1021.70 | 1016.60 | 1021.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 1004.95 | 1014.27 | 1019.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:30:00 | 1018.80 | 1014.27 | 1019.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1034.10 | 1013.41 | 1017.72 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 1035.05 | 1021.27 | 1020.76 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 988.85 | 1015.83 | 1018.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 959.70 | 991.90 | 1004.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 982.40 | 979.83 | 990.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 15:15:00 | 974.70 | 979.83 | 990.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 991.55 | 981.35 | 989.58 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1013.00 | 990.96 | 989.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 1027.70 | 1012.25 | 1003.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1150.25 | 1152.37 | 1136.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 1153.95 | 1152.37 | 1136.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1143.15 | 1155.29 | 1146.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 1140.70 | 1155.29 | 1146.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1146.60 | 1153.55 | 1146.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 1154.30 | 1154.59 | 1147.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:00:00 | 1160.65 | 1156.83 | 1149.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 09:15:00 | 1269.73 | 1217.62 | 1200.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 1232.55 | 1247.59 | 1249.53 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1253.10 | 1249.00 | 1248.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 1256.95 | 1250.59 | 1249.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 1250.00 | 1253.92 | 1251.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 1250.00 | 1253.92 | 1251.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1250.00 | 1253.92 | 1251.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1250.00 | 1253.92 | 1251.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1246.00 | 1252.34 | 1251.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:00:00 | 1246.00 | 1252.34 | 1251.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 1248.70 | 1251.61 | 1250.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:45:00 | 1242.75 | 1251.61 | 1250.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 1248.50 | 1250.35 | 1250.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1235.45 | 1247.37 | 1249.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 1251.55 | 1246.64 | 1247.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 1251.55 | 1246.64 | 1247.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1251.55 | 1246.64 | 1247.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 1251.55 | 1246.64 | 1247.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1252.70 | 1247.85 | 1248.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1252.70 | 1247.85 | 1248.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1250.00 | 1248.28 | 1248.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1225.95 | 1248.28 | 1248.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 1260.00 | 1242.22 | 1243.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 1260.00 | 1242.22 | 1243.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 1245.00 | 1242.77 | 1244.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:30:00 | 1237.55 | 1244.56 | 1244.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 1237.10 | 1244.28 | 1244.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 1255.95 | 1244.15 | 1244.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 1255.95 | 1244.15 | 1244.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 13:15:00 | 1290.40 | 1262.01 | 1253.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 1348.10 | 1413.90 | 1383.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 1348.10 | 1413.90 | 1383.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1348.10 | 1413.90 | 1383.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 1348.10 | 1413.90 | 1383.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1390.65 | 1409.25 | 1383.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 14:15:00 | 1416.35 | 1397.47 | 1384.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-26 09:15:00 | 1557.99 | 1483.47 | 1445.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 1448.85 | 1500.08 | 1502.88 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1624.25 | 1505.26 | 1501.75 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 1497.95 | 1524.61 | 1527.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 13:15:00 | 1493.80 | 1518.45 | 1524.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1479.25 | 1457.92 | 1475.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1479.25 | 1457.92 | 1475.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1479.25 | 1457.92 | 1475.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1486.50 | 1457.92 | 1475.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1483.75 | 1463.08 | 1475.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1483.75 | 1463.08 | 1475.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1483.25 | 1467.12 | 1476.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1491.30 | 1467.12 | 1476.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1476.95 | 1473.46 | 1477.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:30:00 | 1456.95 | 1472.78 | 1476.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 10:15:00 | 1484.90 | 1475.21 | 1477.27 | SL hit (close>static) qty=1.00 sl=1482.90 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1201.10 | 1133.21 | 1128.92 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1120.30 | 1140.96 | 1141.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1053.75 | 1118.52 | 1130.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1087.05 | 1085.83 | 1103.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 1087.05 | 1085.83 | 1103.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1065.25 | 1083.18 | 1099.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 1064.00 | 1077.69 | 1095.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 1125.35 | 1090.45 | 1095.98 | SL hit (close>static) qty=1.00 sl=1101.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 1112.10 | 1098.97 | 1098.94 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1080.90 | 1095.36 | 1097.30 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1112.65 | 1092.99 | 1092.99 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1052.50 | 1088.68 | 1091.72 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 1105.75 | 1082.97 | 1082.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1191.00 | 1111.94 | 1097.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 1139.80 | 1149.51 | 1125.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 1139.80 | 1149.51 | 1125.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1173.15 | 1203.77 | 1185.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 1168.40 | 1203.77 | 1185.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1172.50 | 1197.52 | 1184.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 1175.30 | 1197.52 | 1184.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1160.15 | 1190.05 | 1181.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:45:00 | 1161.40 | 1190.05 | 1181.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 1146.00 | 1176.27 | 1176.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1084.20 | 1150.80 | 1164.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 15:15:00 | 1001.00 | 1000.02 | 1025.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 09:15:00 | 964.35 | 1000.02 | 1025.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 853.90 | 819.54 | 854.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 853.90 | 819.54 | 854.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 845.10 | 824.65 | 853.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 851.95 | 824.65 | 853.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 853.00 | 832.99 | 852.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 853.00 | 832.99 | 852.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 853.10 | 837.01 | 852.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:30:00 | 852.40 | 837.01 | 852.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 851.25 | 839.86 | 852.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:15:00 | 864.00 | 839.86 | 852.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 864.00 | 844.69 | 853.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 867.20 | 844.69 | 853.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 879.20 | 851.59 | 855.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 879.80 | 851.59 | 855.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 882.70 | 857.81 | 858.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 880.15 | 857.81 | 858.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 888.05 | 863.86 | 860.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 10:15:00 | 892.85 | 880.25 | 871.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 874.95 | 894.86 | 884.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 874.95 | 894.86 | 884.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 874.95 | 894.86 | 884.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 874.95 | 894.86 | 884.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 878.20 | 891.53 | 884.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 878.20 | 891.53 | 884.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 886.50 | 890.52 | 884.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:15:00 | 900.25 | 891.56 | 885.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 917.00 | 891.79 | 886.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 12:15:00 | 899.35 | 899.81 | 892.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 862.60 | 888.09 | 889.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 862.60 | 888.09 | 889.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 846.50 | 879.78 | 885.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 09:15:00 | 860.35 | 857.71 | 869.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 10:00:00 | 860.35 | 857.71 | 869.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 878.00 | 861.77 | 870.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:45:00 | 877.40 | 861.77 | 870.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 880.30 | 865.48 | 871.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 12:00:00 | 880.30 | 865.48 | 871.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 872.95 | 866.97 | 871.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:30:00 | 851.85 | 864.91 | 869.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 852.80 | 864.36 | 868.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 884.55 | 870.58 | 869.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 884.55 | 870.58 | 869.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 11:15:00 | 913.00 | 891.45 | 886.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 901.40 | 910.89 | 899.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 901.40 | 910.89 | 899.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 901.40 | 910.89 | 899.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 901.40 | 910.89 | 899.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 916.15 | 911.94 | 901.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 930.10 | 910.64 | 906.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 927.65 | 914.94 | 910.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 927.20 | 917.37 | 912.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 943.00 | 916.28 | 913.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 942.00 | 921.42 | 916.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 953.95 | 933.74 | 927.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-20 11:15:00 | 1023.11 | 990.74 | 966.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 1061.45 | 1089.68 | 1089.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 1056.95 | 1083.13 | 1086.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1039.85 | 1039.44 | 1053.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1039.85 | 1039.44 | 1053.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1039.85 | 1039.44 | 1053.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 1031.95 | 1036.88 | 1048.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 1031.80 | 1034.23 | 1045.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:30:00 | 1020.75 | 1030.08 | 1040.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 1030.25 | 1029.49 | 1036.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1035.00 | 1030.59 | 1036.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1027.90 | 1030.59 | 1036.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 1044.50 | 1033.59 | 1036.90 | SL hit (close>static) qty=1.00 sl=1038.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1043.50 | 1039.14 | 1038.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1071.70 | 1046.04 | 1042.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1042.00 | 1056.80 | 1051.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1042.00 | 1056.80 | 1051.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1042.00 | 1056.80 | 1051.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:15:00 | 1034.85 | 1056.80 | 1051.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1040.00 | 1053.44 | 1050.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1036.40 | 1053.44 | 1050.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 1037.65 | 1046.75 | 1047.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 15:15:00 | 1030.00 | 1041.86 | 1045.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1034.85 | 1001.73 | 1014.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1034.85 | 1001.73 | 1014.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1034.85 | 1001.73 | 1014.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 14:00:00 | 1011.10 | 1013.15 | 1017.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1008.60 | 1016.10 | 1017.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 14:15:00 | 1028.20 | 1018.44 | 1018.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1028.20 | 1018.44 | 1018.13 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 10:15:00 | 1014.30 | 1018.03 | 1018.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 13:15:00 | 1005.90 | 1014.47 | 1016.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1011.30 | 1009.21 | 1013.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 1011.30 | 1009.21 | 1013.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 1011.30 | 1009.21 | 1013.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 10:45:00 | 1007.20 | 1009.17 | 1012.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 12:15:00 | 1007.00 | 1009.48 | 1012.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 13:30:00 | 1007.20 | 1008.95 | 1011.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 1036.00 | 1014.16 | 1013.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 1036.00 | 1014.16 | 1013.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1059.20 | 1027.34 | 1019.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 1248.90 | 1249.97 | 1200.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 1248.90 | 1249.97 | 1200.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1230.90 | 1244.12 | 1232.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 1230.90 | 1244.12 | 1232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1224.00 | 1240.10 | 1231.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 1206.90 | 1240.10 | 1231.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1205.80 | 1233.24 | 1229.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1205.80 | 1233.24 | 1229.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1203.00 | 1227.19 | 1226.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 1203.50 | 1227.19 | 1226.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1217.50 | 1225.25 | 1225.93 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1248.90 | 1227.99 | 1226.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1276.10 | 1244.62 | 1236.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1213.00 | 1253.18 | 1247.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1213.00 | 1253.18 | 1247.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1213.00 | 1253.18 | 1247.73 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1203.50 | 1243.24 | 1243.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1198.00 | 1234.19 | 1239.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1061.40 | 1055.89 | 1087.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 1061.40 | 1055.89 | 1087.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1106.60 | 1066.04 | 1088.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 1106.60 | 1066.04 | 1088.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1100.70 | 1072.97 | 1090.04 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1115.00 | 1100.63 | 1098.77 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 1081.60 | 1095.28 | 1096.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 1070.00 | 1090.23 | 1094.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 1064.60 | 1054.49 | 1068.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 1064.60 | 1054.49 | 1068.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 1064.60 | 1054.49 | 1068.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 1064.60 | 1054.49 | 1068.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1071.00 | 1057.79 | 1068.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1103.00 | 1057.79 | 1068.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1102.30 | 1066.69 | 1071.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 1098.00 | 1066.69 | 1071.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 1097.90 | 1072.97 | 1074.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 1105.20 | 1079.42 | 1077.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1105.20 | 1079.42 | 1077.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1111.30 | 1089.73 | 1082.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1067.00 | 1093.49 | 1086.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1067.00 | 1093.49 | 1086.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1067.00 | 1093.49 | 1086.72 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1062.00 | 1080.79 | 1082.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 14:15:00 | 1047.00 | 1070.51 | 1076.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 1073.90 | 1068.37 | 1074.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1073.90 | 1068.37 | 1074.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1073.90 | 1068.37 | 1074.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 1079.70 | 1068.37 | 1074.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1083.20 | 1071.34 | 1075.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 12:30:00 | 1073.00 | 1072.09 | 1075.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 15:15:00 | 1067.00 | 1058.02 | 1057.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1067.00 | 1058.02 | 1057.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1076.30 | 1061.68 | 1059.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 1073.00 | 1075.70 | 1070.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 11:45:00 | 1072.40 | 1075.70 | 1070.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1062.80 | 1072.66 | 1070.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1062.80 | 1072.66 | 1070.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1065.20 | 1071.17 | 1069.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 1069.80 | 1069.89 | 1069.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1058.00 | 1067.34 | 1068.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1058.00 | 1067.34 | 1068.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 1049.30 | 1059.15 | 1063.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 1053.00 | 1052.57 | 1057.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 1053.00 | 1052.57 | 1057.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1053.00 | 1052.57 | 1057.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 1053.00 | 1052.57 | 1057.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1067.60 | 1055.57 | 1058.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 1068.70 | 1055.57 | 1058.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1069.90 | 1058.44 | 1059.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1069.90 | 1058.44 | 1059.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1075.60 | 1061.87 | 1060.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1080.00 | 1074.32 | 1070.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1075.00 | 1076.50 | 1072.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 1075.00 | 1076.50 | 1072.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1079.50 | 1077.10 | 1073.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1082.40 | 1077.28 | 1073.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:45:00 | 1081.40 | 1077.99 | 1075.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 1071.00 | 1074.04 | 1074.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 1071.00 | 1074.04 | 1074.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 1069.90 | 1073.21 | 1074.01 | Break + close below crossover candle low |

### Cycle 77 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1094.30 | 1076.85 | 1075.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 1112.60 | 1086.52 | 1081.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 13:15:00 | 1135.00 | 1137.10 | 1119.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 1135.00 | 1137.10 | 1119.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1241.30 | 1258.63 | 1245.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1225.60 | 1258.63 | 1245.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1234.60 | 1253.82 | 1244.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1224.10 | 1253.82 | 1244.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1216.00 | 1237.88 | 1238.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1192.30 | 1216.51 | 1226.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1203.10 | 1197.05 | 1209.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 1203.10 | 1197.05 | 1209.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1203.10 | 1197.05 | 1209.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 1205.00 | 1197.05 | 1209.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1203.20 | 1198.28 | 1208.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 1203.90 | 1198.28 | 1208.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1209.00 | 1200.43 | 1208.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1189.50 | 1200.43 | 1208.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1184.00 | 1197.14 | 1206.44 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1224.90 | 1210.10 | 1209.55 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 1208.00 | 1211.65 | 1211.78 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 1239.00 | 1217.12 | 1214.25 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1196.30 | 1213.35 | 1215.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1190.10 | 1203.33 | 1209.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1225.40 | 1207.74 | 1211.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1225.40 | 1207.74 | 1211.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1225.40 | 1207.74 | 1211.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1225.40 | 1207.74 | 1211.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1242.00 | 1214.59 | 1213.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 1256.00 | 1222.87 | 1217.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1329.70 | 1330.65 | 1301.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:30:00 | 1332.00 | 1330.65 | 1301.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1315.00 | 1338.89 | 1327.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 1315.00 | 1338.89 | 1327.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1325.70 | 1336.25 | 1327.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1349.40 | 1329.12 | 1326.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 1337.50 | 1343.54 | 1344.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 1337.50 | 1343.54 | 1344.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 1332.10 | 1341.25 | 1343.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1334.20 | 1326.87 | 1332.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1334.20 | 1326.87 | 1332.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1334.20 | 1326.87 | 1332.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1334.20 | 1326.87 | 1332.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1345.30 | 1330.56 | 1333.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 1345.30 | 1330.56 | 1333.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1341.00 | 1332.65 | 1333.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:45:00 | 1334.10 | 1333.74 | 1334.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 1331.40 | 1332.69 | 1333.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1267.39 | 1286.32 | 1304.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1264.83 | 1286.32 | 1304.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 12:15:00 | 1269.30 | 1265.36 | 1281.32 | SL hit (close>ema200) qty=0.50 sl=1265.36 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 1305.30 | 1286.47 | 1284.83 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 1274.80 | 1290.22 | 1290.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1268.10 | 1285.80 | 1288.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1293.00 | 1280.67 | 1283.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1293.00 | 1280.67 | 1283.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1293.00 | 1280.67 | 1283.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1293.00 | 1280.67 | 1283.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1284.80 | 1281.49 | 1283.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 1281.60 | 1281.49 | 1283.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 1281.80 | 1280.75 | 1283.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1276.00 | 1277.69 | 1280.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1295.10 | 1283.25 | 1282.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1295.10 | 1283.25 | 1282.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1298.90 | 1286.38 | 1284.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1291.90 | 1292.36 | 1288.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 1291.90 | 1292.36 | 1288.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1285.70 | 1291.02 | 1287.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1285.70 | 1291.02 | 1287.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1290.50 | 1290.92 | 1288.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 1290.00 | 1290.92 | 1288.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1287.40 | 1290.22 | 1288.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 1288.00 | 1290.22 | 1288.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1286.90 | 1289.55 | 1288.01 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 1273.70 | 1284.54 | 1285.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 1270.40 | 1280.30 | 1283.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1271.40 | 1265.25 | 1270.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1271.40 | 1265.25 | 1270.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1271.40 | 1265.25 | 1270.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 1271.40 | 1265.25 | 1270.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1269.00 | 1266.00 | 1270.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 1269.00 | 1266.00 | 1270.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1273.40 | 1267.48 | 1270.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1273.40 | 1267.48 | 1270.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1273.90 | 1268.76 | 1270.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 1275.60 | 1268.76 | 1270.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1274.00 | 1270.91 | 1271.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:15:00 | 1274.60 | 1270.91 | 1271.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1277.10 | 1272.74 | 1272.38 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1255.80 | 1271.95 | 1273.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1233.00 | 1256.37 | 1264.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 1093.10 | 1092.93 | 1109.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 11:15:00 | 1106.50 | 1095.96 | 1108.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1106.50 | 1095.96 | 1108.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:45:00 | 1106.90 | 1095.96 | 1108.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1116.30 | 1100.03 | 1108.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1116.30 | 1100.03 | 1108.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1106.30 | 1101.28 | 1108.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 1103.90 | 1101.28 | 1108.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 1104.40 | 1103.65 | 1107.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 1119.00 | 1111.47 | 1110.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 1119.00 | 1111.47 | 1110.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 1125.20 | 1114.21 | 1112.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 1118.90 | 1122.25 | 1118.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 1118.90 | 1122.25 | 1118.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1118.90 | 1122.25 | 1118.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1112.00 | 1120.20 | 1118.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1106.30 | 1117.42 | 1117.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1106.30 | 1117.42 | 1117.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1122.30 | 1118.40 | 1117.49 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 1113.90 | 1116.57 | 1116.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1108.00 | 1114.86 | 1116.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 14:15:00 | 1073.00 | 1070.54 | 1085.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 1073.00 | 1070.54 | 1085.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1086.50 | 1073.73 | 1085.52 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1103.20 | 1088.95 | 1087.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1120.80 | 1094.66 | 1090.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1110.20 | 1111.97 | 1103.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1110.20 | 1111.97 | 1103.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1129.50 | 1123.57 | 1114.61 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1115.40 | 1116.45 | 1116.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 1113.60 | 1115.88 | 1116.23 | Break + close below crossover candle low |

### Cycle 95 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1121.30 | 1116.97 | 1116.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1138.80 | 1126.54 | 1122.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 1127.70 | 1129.98 | 1125.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 1127.70 | 1129.98 | 1125.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1123.00 | 1128.58 | 1125.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1131.20 | 1128.58 | 1125.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1129.60 | 1128.41 | 1125.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1121.80 | 1127.09 | 1125.29 | SL hit (close<static) qty=1.00 sl=1123.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1117.30 | 1123.42 | 1123.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1111.30 | 1120.99 | 1122.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1056.40 | 1040.33 | 1054.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1056.40 | 1040.33 | 1054.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1056.40 | 1040.33 | 1054.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1056.40 | 1040.33 | 1054.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1056.60 | 1043.58 | 1054.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1056.60 | 1043.58 | 1054.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1060.80 | 1047.03 | 1055.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 1062.00 | 1047.03 | 1055.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1076.00 | 1059.71 | 1059.35 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 1056.00 | 1061.42 | 1062.03 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1082.80 | 1063.79 | 1062.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1095.30 | 1084.31 | 1075.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1095.30 | 1096.60 | 1087.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1095.30 | 1096.60 | 1087.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1095.30 | 1096.60 | 1087.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 1092.90 | 1096.60 | 1087.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1111.30 | 1112.26 | 1106.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:00:00 | 1115.00 | 1110.61 | 1107.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1096.90 | 1107.41 | 1107.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 1096.90 | 1107.41 | 1107.63 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1109.50 | 1103.93 | 1103.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1115.30 | 1106.20 | 1104.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 1107.10 | 1107.28 | 1105.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1107.10 | 1107.28 | 1105.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1117.80 | 1109.38 | 1106.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 1119.00 | 1109.38 | 1106.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1108.00 | 1133.78 | 1133.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1108.00 | 1133.78 | 1133.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1096.70 | 1126.37 | 1130.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1079.40 | 1071.58 | 1084.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1079.40 | 1071.58 | 1084.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1079.40 | 1071.58 | 1084.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 1051.80 | 1061.98 | 1067.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:00:00 | 1050.70 | 1057.65 | 1064.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 1054.00 | 1056.97 | 1062.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1080.10 | 1066.31 | 1065.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1080.10 | 1066.31 | 1065.35 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 1060.00 | 1065.89 | 1066.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 1055.90 | 1061.49 | 1062.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1041.90 | 1040.53 | 1047.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 1041.90 | 1040.53 | 1047.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1053.80 | 1043.81 | 1048.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1053.80 | 1043.81 | 1048.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1051.00 | 1045.24 | 1048.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 1046.70 | 1045.60 | 1048.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1063.50 | 1051.23 | 1050.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1063.50 | 1051.23 | 1050.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1067.90 | 1055.97 | 1052.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1123.00 | 1123.65 | 1100.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 1123.00 | 1123.65 | 1100.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1111.80 | 1119.70 | 1106.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 1110.10 | 1119.70 | 1106.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1119.80 | 1118.09 | 1107.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1123.40 | 1118.99 | 1109.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1124.40 | 1124.20 | 1115.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1124.00 | 1124.14 | 1117.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:00:00 | 1126.50 | 1124.61 | 1118.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1121.30 | 1124.14 | 1120.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 1124.60 | 1124.14 | 1120.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1130.90 | 1125.50 | 1121.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1120.70 | 1122.46 | 1122.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1120.70 | 1122.46 | 1122.50 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1131.80 | 1124.35 | 1123.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1150.40 | 1128.98 | 1125.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1137.50 | 1139.29 | 1132.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1137.50 | 1139.29 | 1132.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1144.80 | 1139.72 | 1134.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 1152.80 | 1143.56 | 1136.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1118.20 | 1152.51 | 1155.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 1118.20 | 1152.51 | 1155.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 1116.40 | 1131.48 | 1141.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 1111.50 | 1110.88 | 1120.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 1111.50 | 1110.88 | 1120.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1098.20 | 1104.55 | 1114.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1091.70 | 1103.64 | 1109.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 1112.20 | 1084.86 | 1082.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 1112.20 | 1084.86 | 1082.04 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1078.20 | 1084.53 | 1085.17 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 1092.00 | 1086.79 | 1086.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 1092.80 | 1087.99 | 1086.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 1096.70 | 1096.82 | 1092.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 1096.70 | 1096.82 | 1092.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1106.60 | 1100.72 | 1096.02 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1083.30 | 1093.49 | 1094.15 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 1089.80 | 1089.55 | 1089.52 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 1075.50 | 1086.74 | 1088.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1065.80 | 1079.83 | 1083.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 1073.50 | 1069.89 | 1075.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 1073.50 | 1069.89 | 1075.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1073.50 | 1069.89 | 1075.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 1073.50 | 1069.89 | 1075.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1069.00 | 1056.44 | 1061.11 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1069.00 | 1063.83 | 1063.64 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1057.00 | 1063.54 | 1064.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1054.40 | 1061.71 | 1063.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1076.00 | 1061.85 | 1062.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1076.00 | 1061.85 | 1062.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1076.00 | 1061.85 | 1062.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:15:00 | 1112.50 | 1061.85 | 1062.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1102.10 | 1069.90 | 1065.72 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 1072.30 | 1080.65 | 1080.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 1064.60 | 1072.97 | 1075.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 1069.20 | 1068.20 | 1072.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:45:00 | 1070.60 | 1068.20 | 1072.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1070.00 | 1068.19 | 1071.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1070.40 | 1068.19 | 1071.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1064.60 | 1067.48 | 1070.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 1074.80 | 1067.48 | 1070.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1040.70 | 1042.48 | 1049.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:15:00 | 1036.90 | 1042.48 | 1049.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1058.80 | 1037.94 | 1042.78 | SL hit (close>static) qty=1.00 sl=1051.50 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 1077.20 | 1049.90 | 1047.63 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1047.40 | 1054.36 | 1054.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1034.30 | 1047.49 | 1050.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 1040.30 | 1034.06 | 1038.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 1040.30 | 1034.06 | 1038.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1040.30 | 1034.06 | 1038.13 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1058.00 | 1042.67 | 1041.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 13:15:00 | 1062.80 | 1046.70 | 1043.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1100.70 | 1100.84 | 1091.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 1100.70 | 1100.84 | 1091.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1090.00 | 1098.67 | 1090.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1090.00 | 1098.67 | 1090.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1093.00 | 1097.53 | 1091.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1092.30 | 1097.53 | 1091.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1087.40 | 1095.51 | 1090.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:45:00 | 1101.60 | 1094.42 | 1091.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 1099.50 | 1095.69 | 1092.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 1096.50 | 1096.93 | 1094.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 1098.50 | 1096.93 | 1094.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1093.00 | 1097.16 | 1095.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1095.90 | 1097.16 | 1095.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1093.10 | 1096.35 | 1094.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 1086.90 | 1092.72 | 1093.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1086.90 | 1092.72 | 1093.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 09:15:00 | 1082.00 | 1091.26 | 1092.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1076.90 | 1075.08 | 1081.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 14:45:00 | 1079.30 | 1075.08 | 1081.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1082.00 | 1076.85 | 1080.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 1084.90 | 1076.85 | 1080.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1087.00 | 1078.88 | 1081.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 1087.00 | 1078.88 | 1081.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1085.00 | 1080.10 | 1081.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 1081.10 | 1080.10 | 1081.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1099.90 | 1084.85 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1099.90 | 1084.85 | 1083.24 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1073.50 | 1084.32 | 1084.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1070.50 | 1076.61 | 1080.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1076.00 | 1074.47 | 1077.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 1076.00 | 1074.47 | 1077.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1076.00 | 1074.47 | 1077.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1076.00 | 1074.47 | 1077.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1069.90 | 1072.92 | 1076.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 1066.30 | 1069.58 | 1074.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 1065.80 | 1067.60 | 1072.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1064.20 | 1066.58 | 1071.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 1065.60 | 1065.68 | 1069.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1061.20 | 1064.79 | 1069.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 1059.40 | 1064.17 | 1068.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:15:00 | 1060.00 | 1064.17 | 1068.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 1060.00 | 1063.56 | 1067.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1059.10 | 1062.23 | 1066.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 1062.60 | 1062.30 | 1066.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:45:00 | 1065.60 | 1062.30 | 1066.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1050.00 | 1059.76 | 1064.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1057.30 | 1059.76 | 1064.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1047.60 | 1051.21 | 1056.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 1043.20 | 1047.75 | 1054.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:00:00 | 1042.40 | 1046.68 | 1053.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 1042.50 | 1045.94 | 1052.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1065.60 | 1053.15 | 1053.97 | SL hit (close>static) qty=1.00 sl=1063.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1067.40 | 1056.00 | 1055.19 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1046.60 | 1058.72 | 1059.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 1043.90 | 1055.75 | 1058.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1004.90 | 1004.20 | 1018.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1031.00 | 1004.20 | 1018.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1033.20 | 1010.00 | 1019.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1045.10 | 1010.00 | 1019.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1025.10 | 1013.02 | 1019.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1022.40 | 1015.60 | 1020.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 1042.20 | 1023.89 | 1023.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1042.20 | 1023.89 | 1023.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 1050.50 | 1031.67 | 1027.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 1030.60 | 1032.97 | 1028.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 1030.60 | 1032.97 | 1028.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1025.40 | 1032.07 | 1029.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1025.40 | 1032.07 | 1029.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1016.50 | 1028.95 | 1028.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1016.50 | 1028.95 | 1028.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 1019.50 | 1027.06 | 1027.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1008.30 | 1021.52 | 1024.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1020.00 | 1016.43 | 1020.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 1020.00 | 1016.43 | 1020.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1020.00 | 1016.43 | 1020.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1024.00 | 1016.43 | 1020.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1021.50 | 1017.44 | 1021.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 1013.70 | 1016.31 | 1020.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1014.60 | 1014.81 | 1017.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:30:00 | 1014.20 | 1013.99 | 1016.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1004.90 | 1016.47 | 1017.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1014.90 | 1016.16 | 1016.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 1016.70 | 1016.16 | 1016.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1011.80 | 1014.17 | 1015.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 1014.00 | 1014.17 | 1015.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1016.00 | 1014.53 | 1015.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 1016.00 | 1014.53 | 1015.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1015.30 | 1014.69 | 1015.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 1009.00 | 1014.33 | 1015.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1008.50 | 1011.96 | 1014.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 963.01 | 993.49 | 1003.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 963.87 | 993.49 | 1003.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 963.49 | 993.49 | 1003.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 958.55 | 993.49 | 1003.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 958.07 | 993.49 | 1003.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 984.90 | 984.27 | 995.04 | SL hit (close>ema200) qty=0.50 sl=984.27 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1043.40 | 1007.89 | 1003.51 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 989.00 | 1008.46 | 1011.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 967.90 | 989.72 | 999.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 982.10 | 972.41 | 983.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 982.10 | 972.41 | 983.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 982.10 | 972.41 | 983.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 983.30 | 972.41 | 983.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 988.90 | 975.71 | 983.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 988.90 | 975.71 | 983.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 990.50 | 978.66 | 984.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 990.50 | 978.66 | 984.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 995.10 | 987.63 | 987.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1029.40 | 995.98 | 991.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1016.80 | 1017.66 | 1007.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 1016.80 | 1017.66 | 1007.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 989.90 | 1013.74 | 1010.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 989.90 | 1013.74 | 1010.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 992.50 | 1009.49 | 1009.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 991.50 | 1009.49 | 1009.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 989.30 | 1005.45 | 1007.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 985.40 | 1001.44 | 1005.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1008.10 | 980.17 | 986.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 1008.10 | 980.17 | 986.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1008.10 | 980.17 | 986.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 1003.10 | 980.17 | 986.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1018.50 | 987.83 | 989.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 1016.50 | 987.83 | 989.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 1015.90 | 993.45 | 991.89 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 1016.30 | 1020.36 | 1020.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 1011.50 | 1017.38 | 1019.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1023.50 | 1015.62 | 1017.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1023.50 | 1015.62 | 1017.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1023.50 | 1015.62 | 1017.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1025.50 | 1015.62 | 1017.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1013.50 | 1015.20 | 1017.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 1010.50 | 1013.80 | 1016.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 1009.50 | 1012.94 | 1015.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 959.97 | 978.15 | 988.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 959.02 | 978.15 | 988.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 909.45 | 935.68 | 952.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 948.95 | 921.64 | 919.25 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 902.75 | 926.76 | 928.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 894.00 | 910.45 | 918.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 901.80 | 898.66 | 906.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 901.80 | 898.66 | 906.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 901.80 | 898.66 | 906.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 910.70 | 898.66 | 906.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 900.10 | 896.52 | 902.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 905.45 | 896.52 | 902.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 900.35 | 897.60 | 902.00 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 926.05 | 905.02 | 903.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 939.50 | 911.92 | 906.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 916.75 | 929.57 | 920.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 916.75 | 929.57 | 920.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 916.75 | 929.57 | 920.06 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 909.80 | 916.35 | 916.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 907.00 | 914.48 | 915.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 915.15 | 914.61 | 915.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 915.15 | 914.61 | 915.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 915.15 | 914.61 | 915.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 914.85 | 914.61 | 915.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 916.50 | 914.99 | 915.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 921.60 | 914.99 | 915.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 912.80 | 914.55 | 915.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:15:00 | 917.00 | 914.55 | 915.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 917.00 | 915.04 | 915.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 896.40 | 915.04 | 915.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 911.90 | 900.67 | 901.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 906.00 | 902.63 | 902.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 906.00 | 902.63 | 902.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 934.70 | 909.04 | 905.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 916.55 | 927.88 | 919.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 916.55 | 927.88 | 919.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 916.55 | 927.88 | 919.83 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 896.00 | 913.23 | 915.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 877.00 | 896.11 | 905.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 912.00 | 895.10 | 903.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 912.00 | 895.10 | 903.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 912.00 | 895.10 | 903.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 913.50 | 895.10 | 903.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 906.10 | 897.30 | 903.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 902.85 | 902.59 | 904.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 890.15 | 898.25 | 899.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 902.90 | 899.10 | 899.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 909.45 | 901.17 | 900.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 909.45 | 901.17 | 900.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 912.80 | 903.50 | 901.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 896.10 | 907.43 | 904.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 896.10 | 907.43 | 904.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 896.10 | 907.43 | 904.76 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 896.20 | 902.84 | 902.99 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 909.85 | 903.78 | 903.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 939.00 | 910.83 | 906.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 920.80 | 921.94 | 914.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:00:00 | 920.80 | 921.94 | 914.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 907.50 | 918.44 | 914.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 907.50 | 918.44 | 914.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 908.90 | 916.53 | 913.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 915.00 | 916.23 | 914.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 12:15:00 | 904.00 | 913.78 | 913.11 | SL hit (close<static) qty=1.00 sl=904.30 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 13:15:00 | 897.50 | 910.52 | 911.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 14:15:00 | 894.35 | 907.29 | 910.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 10:15:00 | 913.15 | 906.51 | 908.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 10:15:00 | 913.15 | 906.51 | 908.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 913.15 | 906.51 | 908.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 913.15 | 906.51 | 908.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 915.55 | 908.32 | 909.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 915.55 | 908.32 | 909.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 12:15:00 | 917.95 | 910.24 | 910.21 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 900.10 | 909.12 | 909.87 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 922.20 | 909.96 | 908.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 928.70 | 913.71 | 910.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 968.35 | 968.63 | 954.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:30:00 | 964.10 | 968.63 | 954.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 985.75 | 989.00 | 980.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 986.15 | 989.00 | 980.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 986.40 | 989.09 | 984.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 982.50 | 989.09 | 984.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 982.85 | 987.85 | 984.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 978.10 | 987.85 | 984.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 985.50 | 987.38 | 984.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 981.00 | 987.38 | 984.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 984.00 | 986.70 | 984.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 982.80 | 986.70 | 984.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 986.30 | 986.62 | 984.38 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 973.15 | 982.41 | 982.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 956.80 | 977.29 | 980.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 968.55 | 964.51 | 971.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 968.55 | 964.51 | 971.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 968.55 | 964.51 | 971.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 959.00 | 963.92 | 968.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:00:00 | 960.20 | 963.18 | 968.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 959.00 | 962.61 | 967.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 958.00 | 962.07 | 966.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 968.50 | 959.77 | 962.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 968.50 | 959.77 | 962.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 970.50 | 961.91 | 963.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 970.50 | 961.91 | 963.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-29 14:15:00 | 978.50 | 967.22 | 965.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 978.50 | 967.22 | 965.70 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 935.05 | 959.52 | 962.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 907.10 | 949.04 | 957.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 888.60 | 880.73 | 902.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 11:00:00 | 888.60 | 880.73 | 902.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 893.40 | 883.26 | 901.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 896.95 | 883.26 | 901.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 898.40 | 889.02 | 899.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 898.40 | 889.02 | 899.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 893.75 | 889.97 | 899.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 898.65 | 889.97 | 899.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 896.45 | 891.27 | 898.97 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 912.15 | 901.45 | 901.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 915.45 | 904.25 | 902.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 914.55 | 920.02 | 913.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 914.55 | 920.02 | 913.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 914.55 | 920.02 | 913.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 916.25 | 920.02 | 913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 926.80 | 920.67 | 914.85 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 745.20 | 2024-05-14 12:15:00 | 758.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-14 09:30:00 | 754.45 | 2024-05-14 12:15:00 | 758.45 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-06-11 09:15:00 | 718.50 | 2024-06-14 11:15:00 | 716.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-06-24 15:15:00 | 689.95 | 2024-06-26 13:15:00 | 695.70 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-06-25 11:15:00 | 689.70 | 2024-06-26 13:15:00 | 695.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-25 12:30:00 | 689.20 | 2024-06-26 13:15:00 | 695.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-25 15:15:00 | 687.95 | 2024-06-26 13:15:00 | 695.70 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-01 09:15:00 | 713.75 | 2024-07-10 09:15:00 | 785.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-15 14:30:00 | 753.55 | 2024-07-18 10:15:00 | 768.90 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-07-16 09:45:00 | 754.05 | 2024-07-18 10:15:00 | 768.90 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-07-16 10:45:00 | 754.10 | 2024-07-18 10:15:00 | 768.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-16 14:00:00 | 753.15 | 2024-07-18 10:15:00 | 768.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-07-18 09:15:00 | 752.00 | 2024-07-18 10:15:00 | 768.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-07-23 09:30:00 | 726.15 | 2024-07-25 11:15:00 | 742.25 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-07-23 11:15:00 | 726.80 | 2024-07-25 11:15:00 | 742.25 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-23 13:30:00 | 725.70 | 2024-07-25 11:15:00 | 742.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-07-23 14:45:00 | 727.30 | 2024-07-25 11:15:00 | 742.25 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-07-24 12:15:00 | 731.80 | 2024-07-25 11:15:00 | 742.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest1 | 2024-08-12 11:30:00 | 1023.85 | 2024-08-14 09:15:00 | 989.50 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest1 | 2024-08-12 15:00:00 | 1025.95 | 2024-08-14 09:15:00 | 989.50 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2024-08-13 09:45:00 | 1029.50 | 2024-08-14 09:15:00 | 989.50 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-08-16 09:15:00 | 1014.10 | 2024-08-20 12:15:00 | 997.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-08-20 11:45:00 | 1003.85 | 2024-08-20 12:15:00 | 997.15 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-09-02 11:45:00 | 1032.80 | 2024-09-06 15:15:00 | 981.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 11:45:00 | 1032.80 | 2024-09-09 10:15:00 | 1000.00 | STOP_HIT | 0.50 | 3.18% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1105.50 | 2024-10-18 09:15:00 | 1057.30 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2024-10-15 10:30:00 | 1097.80 | 2024-10-18 09:15:00 | 1057.30 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-10-15 11:45:00 | 1097.95 | 2024-10-18 09:15:00 | 1057.30 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-10-15 15:00:00 | 1103.00 | 2024-10-18 09:15:00 | 1057.30 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2024-10-25 10:15:00 | 955.25 | 2024-10-30 09:15:00 | 974.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-10-28 09:30:00 | 953.70 | 2024-10-30 09:15:00 | 974.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-29 09:45:00 | 952.15 | 2024-10-30 09:15:00 | 974.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-29 12:00:00 | 953.70 | 2024-10-30 09:15:00 | 974.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-11-29 14:30:00 | 1154.30 | 2024-12-05 09:15:00 | 1269.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 10:00:00 | 1160.65 | 2024-12-05 10:15:00 | 1276.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-17 12:30:00 | 1237.55 | 2024-12-18 09:15:00 | 1255.95 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-17 14:15:00 | 1237.10 | 2024-12-18 09:15:00 | 1255.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-23 14:15:00 | 1416.35 | 2024-12-26 09:15:00 | 1557.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-08 09:30:00 | 1456.95 | 2025-01-08 10:15:00 | 1484.90 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-01-08 12:45:00 | 1474.10 | 2025-01-10 09:15:00 | 1400.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 1474.10 | 2025-01-13 09:15:00 | 1326.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-29 12:45:00 | 1064.00 | 2025-01-30 09:15:00 | 1125.35 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest2 | 2025-02-24 13:15:00 | 900.25 | 2025-02-27 09:15:00 | 862.60 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-02-24 15:15:00 | 917.00 | 2025-02-27 09:15:00 | 862.60 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2025-02-25 12:15:00 | 899.35 | 2025-02-27 09:15:00 | 862.60 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-03-03 09:30:00 | 851.85 | 2025-03-04 09:15:00 | 884.55 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-03-03 10:30:00 | 852.80 | 2025-03-04 09:15:00 | 884.55 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-03-12 09:30:00 | 930.10 | 2025-03-20 11:15:00 | 1023.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 09:15:00 | 927.65 | 2025-03-20 11:15:00 | 1020.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 09:45:00 | 927.20 | 2025-03-20 11:15:00 | 1019.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 09:15:00 | 943.00 | 2025-03-20 12:15:00 | 1037.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-19 09:15:00 | 953.95 | 2025-03-20 13:15:00 | 1049.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 1031.95 | 2025-04-02 10:15:00 | 1044.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-03-28 15:00:00 | 1031.80 | 2025-04-02 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-04-01 10:30:00 | 1020.75 | 2025-04-02 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-04-01 15:00:00 | 1030.25 | 2025-04-02 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1027.90 | 2025-04-02 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-04-08 14:00:00 | 1011.10 | 2025-04-09 14:15:00 | 1028.20 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-04-09 10:00:00 | 1008.60 | 2025-04-09 14:15:00 | 1028.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-04-15 10:45:00 | 1007.20 | 2025-04-16 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-04-15 12:15:00 | 1007.00 | 2025-04-16 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-15 13:30:00 | 1007.20 | 2025-04-16 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-05-12 10:15:00 | 1098.00 | 2025-05-12 11:15:00 | 1105.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-05-12 10:45:00 | 1097.90 | 2025-05-12 11:15:00 | 1105.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-05-14 12:30:00 | 1073.00 | 2025-05-16 15:15:00 | 1067.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-05-21 10:00:00 | 1069.80 | 2025-05-21 11:15:00 | 1058.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1082.40 | 2025-05-29 13:15:00 | 1071.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-28 14:45:00 | 1081.40 | 2025-05-29 13:15:00 | 1071.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1349.40 | 2025-07-01 12:15:00 | 1337.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-03 12:45:00 | 1334.10 | 2025-07-07 11:15:00 | 1267.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 13:45:00 | 1331.40 | 2025-07-07 11:15:00 | 1264.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 12:45:00 | 1334.10 | 2025-07-08 12:15:00 | 1269.30 | STOP_HIT | 0.50 | 4.86% |
| SELL | retest2 | 2025-07-03 13:45:00 | 1331.40 | 2025-07-08 12:15:00 | 1269.30 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2025-07-14 11:15:00 | 1281.60 | 2025-07-15 11:15:00 | 1295.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-14 12:30:00 | 1281.80 | 2025-07-15 11:15:00 | 1295.10 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1276.00 | 2025-07-15 11:15:00 | 1295.10 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-01 14:15:00 | 1103.90 | 2025-08-04 13:15:00 | 1119.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-04 09:30:00 | 1104.40 | 2025-08-04 13:15:00 | 1119.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-25 09:15:00 | 1131.20 | 2025-08-25 10:15:00 | 1121.80 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-25 10:15:00 | 1129.60 | 2025-08-25 10:15:00 | 1121.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-12 11:00:00 | 1115.00 | 2025-09-15 09:15:00 | 1096.90 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-09-17 15:15:00 | 1119.00 | 2025-09-22 13:15:00 | 1108.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1051.80 | 2025-10-01 14:15:00 | 1080.10 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-09-30 15:00:00 | 1050.70 | 2025-10-01 14:15:00 | 1080.10 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-10-01 10:00:00 | 1054.00 | 2025-10-01 14:15:00 | 1080.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-10-09 13:30:00 | 1046.70 | 2025-10-10 09:15:00 | 1063.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-10-15 12:15:00 | 1123.40 | 2025-10-20 14:15:00 | 1120.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1124.40 | 2025-10-20 14:15:00 | 1120.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-16 11:00:00 | 1124.00 | 2025-10-20 14:15:00 | 1120.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-16 12:00:00 | 1126.50 | 2025-10-20 14:15:00 | 1120.70 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-24 11:45:00 | 1152.80 | 2025-10-29 10:15:00 | 1118.20 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1091.70 | 2025-11-07 14:15:00 | 1112.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-10 10:15:00 | 1036.90 | 2025-12-11 09:15:00 | 1058.80 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-26 14:45:00 | 1101.60 | 2025-12-30 12:15:00 | 1086.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-12-29 09:30:00 | 1099.50 | 2025-12-30 12:15:00 | 1086.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-29 12:30:00 | 1096.50 | 2025-12-30 12:15:00 | 1086.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-29 13:00:00 | 1098.50 | 2025-12-30 12:15:00 | 1086.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-02 12:15:00 | 1081.10 | 2026-01-05 09:15:00 | 1099.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-08 10:30:00 | 1066.30 | 2026-01-14 10:15:00 | 1065.60 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-01-08 12:30:00 | 1065.80 | 2026-01-14 10:15:00 | 1065.60 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1064.20 | 2026-01-14 10:15:00 | 1065.60 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-01-09 10:00:00 | 1065.60 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-01-09 11:30:00 | 1059.40 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-09 12:15:00 | 1060.00 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-09 12:45:00 | 1060.00 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-09 13:45:00 | 1059.10 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1043.20 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-01-13 13:00:00 | 1042.40 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-13 13:30:00 | 1042.50 | 2026-01-14 11:15:00 | 1067.40 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1022.40 | 2026-01-22 13:15:00 | 1042.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-01-28 10:30:00 | 1013.70 | 2026-02-02 09:15:00 | 963.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1014.60 | 2026-02-02 09:15:00 | 963.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 11:30:00 | 1014.20 | 2026-02-02 09:15:00 | 963.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1004.90 | 2026-02-02 09:15:00 | 958.55 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1009.00 | 2026-02-02 09:15:00 | 958.07 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-01-28 10:30:00 | 1013.70 | 2026-02-02 13:15:00 | 984.90 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1014.60 | 2026-02-02 13:15:00 | 984.90 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-01-29 11:30:00 | 1014.20 | 2026-02-02 13:15:00 | 984.90 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1004.90 | 2026-02-02 13:15:00 | 984.90 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1009.00 | 2026-02-02 13:15:00 | 984.90 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2026-02-01 11:30:00 | 1008.50 | 2026-02-03 10:15:00 | 1043.40 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1010.50 | 2026-02-27 10:15:00 | 959.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1009.50 | 2026-02-27 10:15:00 | 959.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1010.50 | 2026-03-04 09:15:00 | 909.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1009.50 | 2026-03-04 09:15:00 | 908.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 896.40 | 2026-03-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-24 13:45:00 | 911.90 | 2026-03-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2026-04-01 14:45:00 | 902.85 | 2026-04-06 11:15:00 | 909.45 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-04-06 09:15:00 | 890.15 | 2026-04-06 11:15:00 | 909.45 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-04-06 11:00:00 | 902.90 | 2026-04-06 11:15:00 | 909.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-09 12:00:00 | 915.00 | 2026-04-09 12:15:00 | 904.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-27 14:00:00 | 959.00 | 2026-04-29 14:15:00 | 978.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-04-27 15:00:00 | 960.20 | 2026-04-29 14:15:00 | 978.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-04-28 09:15:00 | 959.00 | 2026-04-29 14:15:00 | 978.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-04-28 10:30:00 | 958.00 | 2026-04-29 14:15:00 | 978.50 | STOP_HIT | 1.00 | -2.14% |
