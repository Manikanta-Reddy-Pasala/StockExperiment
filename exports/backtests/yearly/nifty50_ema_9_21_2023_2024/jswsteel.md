# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (2753 bars)
- **Last close:** 1272.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 87 |
| ALERT2 | 85 |
| ALERT2_SKIP | 50 |
| ALERT3 | 212 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 91 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 90 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 16 / 76
- **Target hits / Stop hits / Partials:** 0 / 89 / 3
- **Avg / median % per leg:** -1.23% / -0.81%
- **Sum % (uncompounded):** -112.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 8 | 15.4% | 0 | 52 | 0 | -0.49% | -25.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| BUY @ 3rd Alert (retest2) | 51 | 8 | 15.7% | 0 | 51 | 0 | -0.47% | -24.2% |
| SELL (all) | 40 | 8 | 20.0% | 0 | 37 | 3 | -2.19% | -87.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 8 | 20.0% | 0 | 37 | 3 | -2.19% | -87.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| retest2 (combined) | 91 | 16 | 17.6% | 0 | 88 | 3 | -1.23% | -111.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 704.10 | 695.92 | 695.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 709.00 | 701.51 | 698.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 704.30 | 704.40 | 700.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 14:45:00 | 704.90 | 704.40 | 700.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 704.30 | 704.23 | 701.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 11:15:00 | 705.50 | 704.29 | 701.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 14:15:00 | 696.40 | 701.41 | 701.09 | SL hit (close<static) qty=1.00 sl=697.90 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 696.50 | 700.43 | 700.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 692.95 | 698.93 | 699.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 696.70 | 695.57 | 697.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-25 15:00:00 | 696.70 | 695.57 | 697.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 695.55 | 695.57 | 697.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 697.10 | 695.57 | 697.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 698.90 | 696.23 | 697.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:00:00 | 698.90 | 696.23 | 697.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 704.95 | 697.98 | 698.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:00:00 | 704.95 | 697.98 | 698.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 700.75 | 698.61 | 698.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 702.40 | 699.37 | 698.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 702.60 | 703.46 | 701.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 702.60 | 703.46 | 701.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 702.60 | 703.46 | 701.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:30:00 | 708.55 | 706.08 | 703.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 15:15:00 | 705.30 | 706.64 | 704.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 09:15:00 | 697.70 | 704.64 | 704.15 | SL hit (close<static) qty=1.00 sl=701.10 alert=retest2 |

### Cycle 4 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 698.45 | 703.40 | 703.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 692.95 | 701.31 | 702.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 15:15:00 | 700.00 | 698.65 | 700.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 15:15:00 | 700.00 | 698.65 | 700.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 700.00 | 698.65 | 700.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 693.90 | 698.65 | 700.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 695.85 | 698.09 | 700.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 12:00:00 | 692.45 | 696.47 | 699.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 15:15:00 | 692.95 | 696.04 | 698.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 14:15:00 | 706.50 | 698.81 | 698.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 14:15:00 | 706.50 | 698.81 | 698.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 14:15:00 | 711.90 | 705.53 | 702.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 14:15:00 | 748.00 | 748.15 | 740.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 15:00:00 | 748.00 | 748.15 | 740.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 746.90 | 749.25 | 744.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 757.20 | 748.89 | 745.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 09:15:00 | 760.25 | 771.06 | 771.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 09:15:00 | 760.25 | 771.06 | 771.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 10:15:00 | 754.15 | 767.68 | 770.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 748.35 | 744.71 | 749.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 748.35 | 744.71 | 749.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 748.35 | 744.71 | 749.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 748.35 | 744.71 | 749.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 749.50 | 745.67 | 749.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 756.00 | 745.67 | 749.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 755.60 | 747.65 | 749.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:15:00 | 759.60 | 747.65 | 749.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 755.00 | 749.12 | 750.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 755.35 | 749.12 | 750.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 755.95 | 751.89 | 751.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 762.50 | 754.02 | 752.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 792.55 | 792.67 | 786.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 10:00:00 | 792.55 | 792.67 | 786.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 792.35 | 794.80 | 791.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:45:00 | 792.00 | 794.80 | 791.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 793.05 | 794.45 | 792.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:30:00 | 792.00 | 794.45 | 792.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 792.00 | 793.96 | 792.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 14:45:00 | 788.35 | 793.96 | 792.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 793.50 | 793.87 | 792.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 796.85 | 793.87 | 792.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 793.45 | 793.78 | 792.26 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 789.00 | 792.40 | 792.69 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 807.95 | 794.11 | 793.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 794.30 | 800.98 | 801.33 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 805.70 | 801.44 | 801.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 808.40 | 802.83 | 801.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 11:15:00 | 802.35 | 803.38 | 802.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 11:15:00 | 802.35 | 803.38 | 802.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 802.35 | 803.38 | 802.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:00:00 | 802.35 | 803.38 | 802.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 801.70 | 803.04 | 802.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:45:00 | 800.95 | 803.04 | 802.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 799.65 | 802.36 | 802.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 799.95 | 802.36 | 802.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 14:15:00 | 797.80 | 801.45 | 801.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 09:15:00 | 796.75 | 800.03 | 800.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 11:15:00 | 794.95 | 793.97 | 796.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 11:15:00 | 794.95 | 793.97 | 796.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 794.95 | 793.97 | 796.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:00:00 | 794.95 | 793.97 | 796.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 791.25 | 792.34 | 794.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:00:00 | 788.65 | 791.61 | 793.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:45:00 | 789.45 | 791.22 | 793.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 14:15:00 | 798.80 | 793.17 | 793.88 | SL hit (close>static) qty=1.00 sl=795.90 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 800.65 | 795.59 | 794.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 807.85 | 797.97 | 796.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 784.20 | 797.83 | 796.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 14:15:00 | 784.20 | 797.83 | 796.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 784.20 | 797.83 | 796.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 784.20 | 797.83 | 796.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 15:15:00 | 788.10 | 795.88 | 795.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 09:15:00 | 778.25 | 792.36 | 794.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 792.50 | 782.96 | 787.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 792.50 | 782.96 | 787.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 792.50 | 782.96 | 787.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 791.90 | 782.96 | 787.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 795.35 | 785.44 | 787.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 795.00 | 785.44 | 787.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 12:15:00 | 801.80 | 790.79 | 790.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 13:15:00 | 803.00 | 793.23 | 791.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 10:15:00 | 801.75 | 803.74 | 799.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 10:45:00 | 801.40 | 803.74 | 799.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 798.60 | 802.72 | 799.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:45:00 | 799.75 | 802.72 | 799.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 797.65 | 801.70 | 799.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:00:00 | 797.65 | 801.70 | 799.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 801.15 | 801.59 | 799.79 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 795.35 | 798.56 | 798.69 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 12:15:00 | 802.95 | 799.09 | 798.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 815.55 | 804.15 | 801.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 811.75 | 817.43 | 813.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 811.75 | 817.43 | 813.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 811.75 | 817.43 | 813.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 811.15 | 817.43 | 813.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 809.95 | 815.93 | 813.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 809.95 | 815.93 | 813.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 809.35 | 814.61 | 812.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 809.00 | 814.61 | 812.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 802.30 | 811.61 | 811.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 797.45 | 806.37 | 809.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 11:15:00 | 806.30 | 806.09 | 808.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 12:00:00 | 806.30 | 806.09 | 808.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 808.00 | 804.82 | 806.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 811.30 | 804.82 | 806.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 808.40 | 805.53 | 807.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:45:00 | 805.75 | 805.39 | 806.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 805.90 | 806.36 | 806.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 09:45:00 | 806.00 | 806.15 | 806.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 12:15:00 | 813.75 | 808.11 | 807.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 12:15:00 | 813.75 | 808.11 | 807.52 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 11:15:00 | 803.20 | 807.82 | 807.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 15:15:00 | 800.00 | 804.47 | 806.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 10:15:00 | 808.00 | 805.02 | 806.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 10:15:00 | 808.00 | 805.02 | 806.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 808.00 | 805.02 | 806.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:00:00 | 808.00 | 805.02 | 806.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 809.55 | 805.93 | 806.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:30:00 | 809.85 | 805.93 | 806.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 12:15:00 | 811.50 | 807.04 | 806.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 14:15:00 | 824.00 | 811.10 | 808.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 817.45 | 824.47 | 819.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 817.45 | 824.47 | 819.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 817.45 | 824.47 | 819.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 817.45 | 824.47 | 819.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 822.20 | 824.01 | 819.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 11:15:00 | 823.25 | 824.01 | 819.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 11:45:00 | 823.05 | 823.82 | 819.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 14:00:00 | 822.95 | 823.44 | 820.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 09:15:00 | 802.00 | 818.42 | 818.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 802.00 | 818.42 | 818.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 10:15:00 | 795.50 | 813.83 | 816.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 13:15:00 | 791.80 | 788.54 | 792.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 13:15:00 | 791.80 | 788.54 | 792.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 791.80 | 788.54 | 792.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:00:00 | 791.80 | 788.54 | 792.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 786.55 | 788.14 | 791.86 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 794.90 | 792.82 | 792.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 795.90 | 793.67 | 793.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 796.50 | 796.80 | 795.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:15:00 | 803.50 | 796.80 | 795.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 795.00 | 797.73 | 796.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-24 11:15:00 | 795.00 | 797.73 | 796.43 | SL hit (close<ema400) qty=1.00 sl=796.43 alert=retest1 |

### Cycle 24 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 788.85 | 794.50 | 795.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 785.05 | 791.64 | 793.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 779.45 | 776.20 | 780.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 10:15:00 | 778.45 | 776.20 | 780.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 782.30 | 777.42 | 780.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:45:00 | 782.80 | 777.42 | 780.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 782.40 | 778.41 | 780.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 780.40 | 780.73 | 781.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 09:15:00 | 787.80 | 782.43 | 781.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 787.80 | 782.43 | 781.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 10:15:00 | 790.70 | 784.08 | 782.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 785.10 | 785.61 | 783.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 14:00:00 | 785.10 | 785.61 | 783.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 788.00 | 786.09 | 784.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:45:00 | 787.20 | 786.09 | 784.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 788.40 | 786.71 | 785.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:45:00 | 786.05 | 786.71 | 785.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 787.25 | 786.84 | 785.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 13:15:00 | 785.35 | 786.84 | 785.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 782.90 | 786.05 | 785.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:00:00 | 782.90 | 786.05 | 785.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 782.10 | 785.26 | 784.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 782.10 | 785.26 | 784.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 780.35 | 784.28 | 784.45 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 799.20 | 787.26 | 785.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 806.20 | 795.28 | 790.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 09:15:00 | 802.75 | 807.52 | 802.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 802.75 | 807.52 | 802.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 802.75 | 807.52 | 802.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 14:15:00 | 809.95 | 806.76 | 803.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 12:15:00 | 810.50 | 809.93 | 806.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 13:30:00 | 809.70 | 809.51 | 806.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:15:00 | 812.65 | 809.51 | 806.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 810.50 | 810.93 | 808.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 10:15:00 | 814.85 | 810.93 | 808.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 11:45:00 | 812.30 | 811.80 | 809.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 13:15:00 | 814.40 | 811.57 | 809.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 11:00:00 | 812.70 | 814.71 | 812.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 11:15:00 | 818.60 | 815.49 | 812.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 10:00:00 | 820.25 | 816.76 | 814.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 13:00:00 | 819.20 | 821.14 | 819.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 14:00:00 | 820.25 | 820.96 | 819.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 09:15:00 | 820.00 | 818.93 | 818.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 09:15:00 | 809.40 | 817.02 | 817.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 809.40 | 817.02 | 817.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 10:15:00 | 806.55 | 811.29 | 813.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 14:15:00 | 813.40 | 810.34 | 812.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 813.40 | 810.34 | 812.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 813.40 | 810.34 | 812.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 813.40 | 810.34 | 812.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 812.25 | 810.72 | 812.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 811.95 | 810.72 | 812.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 811.45 | 810.94 | 812.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 11:30:00 | 809.70 | 810.49 | 811.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 12:45:00 | 808.85 | 810.14 | 811.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 14:45:00 | 809.60 | 809.94 | 811.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 808.00 | 809.73 | 810.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 983.35 | 844.45 | 826.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 983.35 | 844.45 | 826.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 994.65 | 897.29 | 855.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 995.80 | 996.42 | 959.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:30:00 | 996.20 | 996.42 | 959.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1033.60 | 1038.63 | 1028.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1031.30 | 1038.63 | 1028.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1033.40 | 1037.04 | 1032.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:30:00 | 1041.30 | 1036.69 | 1033.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:45:00 | 1042.50 | 1037.89 | 1033.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1020.50 | 1034.64 | 1033.91 | SL hit (close<static) qty=1.00 sl=1030.60 alert=retest2 |

### Cycle 30 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1025.90 | 1032.89 | 1033.18 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 1046.90 | 1035.11 | 1033.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 1056.50 | 1039.39 | 1035.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 1043.20 | 1050.42 | 1044.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 1042.80 | 1048.90 | 1044.54 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1031.70 | 1040.38 | 1041.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 1028.00 | 1035.66 | 1038.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 1036.10 | 1033.11 | 1036.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1003.80 | 1027.25 | 1033.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 988.40 | 1027.25 | 1033.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 938.98 | 956.26 | 963.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 958.30 | 955.12 | 961.37 | SL hit (close>ema200) qty=0.50 sl=955.12 alert=retest2 |

### Cycle 33 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 987.50 | 966.36 | 964.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1000.90 | 976.56 | 969.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 991.90 | 991.99 | 981.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 991.90 | 991.99 | 981.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 989.80 | 995.02 | 989.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 13:00:00 | 989.80 | 995.02 | 989.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 991.90 | 994.40 | 989.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1010.50 | 992.55 | 989.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 1013.70 | 1018.34 | 1018.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 1013.70 | 1018.34 | 1018.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1003.60 | 1014.50 | 1016.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 1003.10 | 1002.97 | 1008.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 1003.10 | 1002.97 | 1008.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1005.70 | 1001.56 | 1006.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 1005.70 | 1001.56 | 1006.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1008.00 | 1002.85 | 1006.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1014.10 | 1002.85 | 1006.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1013.70 | 1005.02 | 1007.22 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1011.40 | 1009.07 | 1008.81 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 1005.90 | 1008.50 | 1008.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1000.50 | 1006.90 | 1007.93 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1034.00 | 1010.96 | 1009.51 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 1003.00 | 1013.05 | 1014.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 1000.40 | 1010.52 | 1013.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 1010.00 | 1008.09 | 1010.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1003.40 | 977.85 | 976.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1003.40 | 977.85 | 976.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 1008.30 | 983.94 | 979.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 1003.40 | 1007.02 | 1000.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 1003.40 | 1007.02 | 1000.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1009.85 | 1007.58 | 1001.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1013.50 | 1005.85 | 1001.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:45:00 | 1011.55 | 1006.89 | 1002.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 999.85 | 1004.94 | 1004.53 | SL hit (close<static) qty=1.00 sl=1000.80 alert=retest2 |

### Cycle 40 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 997.25 | 1003.40 | 1003.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 989.00 | 998.47 | 1001.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 996.05 | 990.87 | 994.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 996.05 | 990.87 | 994.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 996.05 | 990.87 | 994.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 996.05 | 990.87 | 994.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 999.70 | 992.64 | 994.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 998.55 | 992.64 | 994.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1008.00 | 997.36 | 996.81 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 993.00 | 998.16 | 998.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 988.95 | 995.31 | 997.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 993.00 | 991.51 | 994.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:45:00 | 993.25 | 991.51 | 994.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 995.25 | 992.25 | 994.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 995.25 | 992.25 | 994.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 994.40 | 992.68 | 994.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 990.45 | 992.34 | 994.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:00:00 | 990.95 | 992.34 | 994.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 996.90 | 993.51 | 994.40 | SL hit (close>static) qty=1.00 sl=995.65 alert=retest2 |

### Cycle 43 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1022.00 | 1000.03 | 997.17 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 998.65 | 999.23 | 999.25 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1002.80 | 999.94 | 999.57 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 996.50 | 999.08 | 999.27 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1001.70 | 999.60 | 999.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 1016.55 | 1002.99 | 1001.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1005.75 | 1007.97 | 1004.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1005.75 | 1007.97 | 1004.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1027.00 | 1028.02 | 1023.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 1025.35 | 1028.02 | 1023.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1024.85 | 1027.49 | 1024.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1024.35 | 1027.49 | 1024.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1023.15 | 1026.62 | 1024.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 1022.25 | 1026.62 | 1024.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1024.65 | 1026.23 | 1024.32 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 1016.55 | 1022.76 | 1022.98 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 1026.60 | 1022.60 | 1022.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 1030.90 | 1024.26 | 1023.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 1051.60 | 1052.39 | 1043.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 1051.60 | 1052.39 | 1043.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1046.00 | 1050.12 | 1044.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 1044.80 | 1050.12 | 1044.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1044.00 | 1048.33 | 1044.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 1043.00 | 1048.33 | 1044.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1041.90 | 1047.05 | 1044.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 1041.90 | 1047.05 | 1044.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1034.90 | 1044.62 | 1043.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 1034.90 | 1044.62 | 1043.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1036.20 | 1041.41 | 1041.97 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 1045.00 | 1042.05 | 1041.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 1048.70 | 1043.74 | 1042.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 15:15:00 | 1042.80 | 1043.81 | 1043.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 15:15:00 | 1042.80 | 1043.81 | 1043.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1042.80 | 1043.81 | 1043.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1038.50 | 1043.81 | 1043.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1038.00 | 1042.64 | 1042.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1039.20 | 1042.64 | 1042.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1045.40 | 1043.20 | 1042.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1047.20 | 1043.20 | 1042.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 1038.50 | 1042.26 | 1042.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1038.50 | 1042.26 | 1042.46 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1043.20 | 1041.40 | 1041.38 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1035.40 | 1040.76 | 1041.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 1032.90 | 1037.52 | 1038.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1038.00 | 1037.62 | 1038.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 1038.00 | 1037.62 | 1038.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1038.00 | 1037.62 | 1038.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1038.00 | 1037.62 | 1038.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1040.40 | 1038.17 | 1038.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1040.40 | 1038.17 | 1038.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1035.30 | 1037.60 | 1038.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 1036.40 | 1037.60 | 1038.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1030.60 | 1029.22 | 1032.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 1032.20 | 1029.22 | 1032.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1029.30 | 1028.82 | 1031.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 1030.80 | 1028.82 | 1031.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1032.70 | 1029.59 | 1031.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1032.90 | 1029.59 | 1031.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1039.20 | 1031.51 | 1032.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 1037.50 | 1031.51 | 1032.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 1037.50 | 1033.21 | 1032.87 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1028.10 | 1032.46 | 1032.60 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 1044.80 | 1034.57 | 1033.42 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1031.60 | 1032.56 | 1032.62 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 1036.00 | 1032.87 | 1032.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 1040.70 | 1034.43 | 1033.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 1034.40 | 1034.71 | 1033.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 1034.40 | 1034.71 | 1033.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1033.60 | 1034.48 | 1033.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1032.00 | 1034.48 | 1033.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1031.10 | 1033.81 | 1033.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1031.10 | 1033.81 | 1033.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1029.90 | 1033.03 | 1033.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 1028.40 | 1032.00 | 1032.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1033.10 | 1031.62 | 1032.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 1036.90 | 1032.68 | 1032.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1040.90 | 1034.32 | 1033.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1036.70 | 1036.88 | 1035.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 1036.70 | 1036.88 | 1035.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1034.50 | 1036.41 | 1035.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1034.50 | 1036.41 | 1035.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1035.90 | 1036.31 | 1035.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1033.50 | 1036.31 | 1035.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1029.20 | 1034.88 | 1034.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 1024.80 | 1034.88 | 1034.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1022.30 | 1032.37 | 1033.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 1021.20 | 1026.78 | 1029.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 12:15:00 | 1025.70 | 1025.40 | 1028.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 1025.70 | 1025.40 | 1028.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1028.90 | 1026.10 | 1028.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:45:00 | 1029.40 | 1026.10 | 1028.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1029.10 | 1026.70 | 1028.50 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 1045.80 | 1031.05 | 1030.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1052.40 | 1043.69 | 1039.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1028.80 | 1042.67 | 1040.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1036.90 | 1041.51 | 1040.33 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 1031.90 | 1038.18 | 1038.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1030.40 | 1036.62 | 1038.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1045.40 | 1035.55 | 1037.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1045.40 | 1035.55 | 1037.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1045.40 | 1035.55 | 1037.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 1045.40 | 1035.55 | 1037.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1043.40 | 1037.12 | 1037.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 1046.90 | 1037.12 | 1037.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 1042.90 | 1038.28 | 1038.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 1051.80 | 1040.98 | 1039.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 1049.80 | 1050.55 | 1046.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:45:00 | 1050.00 | 1050.55 | 1046.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1048.80 | 1051.65 | 1048.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 1047.60 | 1051.65 | 1048.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1051.20 | 1051.56 | 1048.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:45:00 | 1057.70 | 1052.70 | 1050.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 13:30:00 | 1057.00 | 1053.66 | 1051.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 1057.50 | 1053.66 | 1051.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 1042.70 | 1055.32 | 1054.25 | SL hit (close<static) qty=1.00 sl=1045.80 alert=retest2 |

### Cycle 66 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 1044.60 | 1053.17 | 1053.37 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1057.60 | 1053.54 | 1053.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1059.60 | 1054.75 | 1053.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1053.80 | 1054.56 | 1053.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1053.50 | 1054.35 | 1053.76 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 1047.40 | 1052.74 | 1053.21 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1053.80 | 1053.15 | 1053.11 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1046.90 | 1052.44 | 1052.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 1041.30 | 1048.20 | 1050.55 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1071.50 | 1052.11 | 1051.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 1079.60 | 1065.24 | 1058.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 12:15:00 | 1071.90 | 1071.97 | 1065.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:45:00 | 1073.00 | 1071.97 | 1065.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1076.70 | 1073.70 | 1068.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 1077.60 | 1073.70 | 1068.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:45:00 | 1076.90 | 1081.65 | 1077.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 1057.00 | 1073.52 | 1074.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 1057.00 | 1073.52 | 1074.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 1054.50 | 1069.72 | 1072.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1059.10 | 1058.71 | 1064.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 1059.10 | 1058.71 | 1064.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1056.70 | 1058.46 | 1062.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1049.70 | 1058.54 | 1062.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1047.80 | 1038.83 | 1038.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1047.80 | 1038.83 | 1038.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1058.60 | 1045.42 | 1042.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1065.10 | 1070.08 | 1063.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1065.10 | 1070.08 | 1063.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1065.10 | 1070.08 | 1063.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1065.10 | 1070.08 | 1063.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1067.90 | 1069.64 | 1064.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 1070.00 | 1068.33 | 1064.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 1093.60 | 1098.46 | 1098.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 1093.60 | 1098.46 | 1098.59 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 1099.50 | 1098.47 | 1098.45 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 1094.90 | 1097.76 | 1098.13 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 1100.50 | 1098.73 | 1098.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1109.60 | 1101.77 | 1100.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 1112.80 | 1112.82 | 1108.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 1112.80 | 1112.82 | 1108.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1108.90 | 1112.37 | 1108.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 1108.90 | 1112.37 | 1108.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1110.80 | 1112.06 | 1109.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 1113.80 | 1111.43 | 1109.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 1133.50 | 1141.48 | 1141.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 1133.50 | 1141.48 | 1141.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 1131.40 | 1139.47 | 1140.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1138.60 | 1137.54 | 1139.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1138.60 | 1137.54 | 1139.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1138.60 | 1137.54 | 1139.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1140.80 | 1137.54 | 1139.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1134.50 | 1136.93 | 1138.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 1136.10 | 1136.93 | 1138.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1139.70 | 1133.03 | 1135.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1139.70 | 1133.03 | 1135.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1141.00 | 1134.62 | 1136.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 1142.00 | 1134.62 | 1136.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 1147.20 | 1139.17 | 1138.07 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 1135.50 | 1138.20 | 1138.40 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1142.80 | 1139.12 | 1138.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1145.90 | 1140.48 | 1139.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1153.60 | 1157.21 | 1151.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 1153.60 | 1157.21 | 1151.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1153.60 | 1157.21 | 1151.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 1153.60 | 1157.21 | 1151.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1154.20 | 1156.61 | 1152.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:45:00 | 1155.90 | 1156.41 | 1152.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 1155.20 | 1159.19 | 1157.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 1148.60 | 1157.07 | 1157.03 | SL hit (close<static) qty=1.00 sl=1149.20 alert=retest2 |

### Cycle 82 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 1152.40 | 1156.14 | 1156.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1142.80 | 1151.84 | 1154.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1159.20 | 1152.42 | 1154.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1167.90 | 1155.51 | 1155.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1170.30 | 1158.47 | 1156.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1159.60 | 1162.00 | 1162.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 1154.10 | 1160.36 | 1161.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1154.40 | 1150.99 | 1155.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1154.40 | 1150.99 | 1155.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1154.40 | 1150.99 | 1155.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1154.40 | 1150.99 | 1155.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1159.60 | 1152.71 | 1155.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1159.60 | 1152.71 | 1155.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1162.00 | 1154.57 | 1156.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 1162.20 | 1154.57 | 1156.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1160.40 | 1157.61 | 1157.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 1165.40 | 1160.46 | 1158.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1165.40 | 1169.88 | 1166.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 1165.40 | 1169.88 | 1166.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1165.40 | 1169.88 | 1166.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 1166.00 | 1169.88 | 1166.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1160.80 | 1168.06 | 1165.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1161.70 | 1168.06 | 1165.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1162.20 | 1166.89 | 1165.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1156.90 | 1166.89 | 1165.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1153.90 | 1163.99 | 1164.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 1146.00 | 1157.25 | 1160.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 14:15:00 | 1153.00 | 1152.89 | 1157.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:15:00 | 1153.30 | 1152.89 | 1157.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1153.40 | 1152.99 | 1156.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 1143.00 | 1151.37 | 1155.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 1143.70 | 1147.90 | 1152.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:45:00 | 1143.70 | 1145.60 | 1151.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 1141.80 | 1144.51 | 1149.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1154.10 | 1143.68 | 1146.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 1156.00 | 1143.68 | 1146.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1146.20 | 1144.18 | 1146.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1145.00 | 1144.18 | 1146.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 1145.00 | 1145.63 | 1146.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:30:00 | 1145.00 | 1146.42 | 1146.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 1152.40 | 1147.62 | 1147.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1152.40 | 1147.62 | 1147.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 1157.50 | 1149.62 | 1148.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 1204.30 | 1205.17 | 1192.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 1204.30 | 1205.17 | 1192.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1204.50 | 1207.47 | 1202.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1204.50 | 1207.47 | 1202.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1206.40 | 1207.25 | 1202.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 1206.40 | 1207.25 | 1202.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1205.80 | 1206.96 | 1203.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 1202.00 | 1206.96 | 1203.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1206.30 | 1206.83 | 1203.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 1207.40 | 1206.83 | 1203.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1195.50 | 1204.56 | 1202.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1195.50 | 1204.56 | 1202.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1196.40 | 1202.93 | 1202.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:45:00 | 1194.00 | 1202.93 | 1202.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 1192.80 | 1200.91 | 1201.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1182.60 | 1196.36 | 1199.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1180.70 | 1169.34 | 1175.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1170.00 | 1169.47 | 1175.36 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1183.30 | 1176.68 | 1176.48 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1170.60 | 1175.63 | 1176.14 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 1199.80 | 1180.39 | 1178.15 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1169.80 | 1182.31 | 1183.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 1166.90 | 1177.42 | 1180.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1172.80 | 1171.38 | 1175.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1172.80 | 1171.38 | 1175.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1172.80 | 1171.38 | 1175.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 1168.20 | 1171.48 | 1175.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:30:00 | 1167.60 | 1170.54 | 1173.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1166.60 | 1167.43 | 1167.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 1170.00 | 1168.03 | 1168.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 1171.30 | 1168.69 | 1168.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1171.30 | 1168.69 | 1168.39 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 1157.10 | 1166.26 | 1167.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 1149.80 | 1159.17 | 1163.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 1120.00 | 1119.82 | 1132.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 1120.00 | 1119.82 | 1132.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1144.20 | 1121.55 | 1128.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1144.20 | 1121.55 | 1128.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1149.30 | 1127.10 | 1130.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1150.30 | 1127.10 | 1130.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1152.00 | 1135.84 | 1133.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1154.00 | 1139.47 | 1135.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1142.10 | 1145.75 | 1141.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 1142.10 | 1145.75 | 1141.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1142.10 | 1145.75 | 1141.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1142.10 | 1145.75 | 1141.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1151.30 | 1146.86 | 1142.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1154.30 | 1146.86 | 1142.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 1151.70 | 1158.26 | 1159.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1151.70 | 1158.26 | 1159.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 1141.70 | 1155.53 | 1157.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 1149.80 | 1149.69 | 1152.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 1149.80 | 1149.69 | 1152.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1155.50 | 1150.85 | 1152.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 1155.20 | 1150.85 | 1152.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1158.30 | 1152.34 | 1153.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 1159.60 | 1152.34 | 1153.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1160.70 | 1154.98 | 1154.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1168.60 | 1157.71 | 1155.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1144.00 | 1154.96 | 1154.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1146.00 | 1153.17 | 1153.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1138.40 | 1150.22 | 1152.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1104.40 | 1102.64 | 1113.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:15:00 | 1107.00 | 1102.64 | 1113.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1116.90 | 1106.00 | 1109.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1118.50 | 1106.00 | 1109.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1118.10 | 1108.42 | 1110.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1118.10 | 1108.42 | 1110.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 1128.70 | 1114.97 | 1113.21 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1099.90 | 1113.10 | 1114.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1096.00 | 1107.62 | 1111.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1087.20 | 1082.94 | 1090.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 1087.20 | 1082.94 | 1090.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1088.30 | 1084.66 | 1090.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 1089.20 | 1084.66 | 1090.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1083.50 | 1084.43 | 1089.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 1087.50 | 1084.43 | 1089.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1085.40 | 1083.88 | 1087.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 1080.50 | 1083.44 | 1087.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 1080.50 | 1082.68 | 1086.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1092.30 | 1082.75 | 1084.88 | SL hit (close>static) qty=1.00 sl=1091.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1095.80 | 1087.05 | 1086.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 1100.70 | 1095.40 | 1091.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1094.80 | 1096.32 | 1093.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 1094.80 | 1096.32 | 1093.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1094.80 | 1096.32 | 1093.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1095.90 | 1096.32 | 1093.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1092.00 | 1095.14 | 1093.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 1092.20 | 1095.14 | 1093.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1091.60 | 1094.44 | 1093.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1090.70 | 1094.44 | 1093.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1093.50 | 1094.25 | 1093.27 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1082.90 | 1090.79 | 1091.87 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1095.70 | 1092.93 | 1092.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 09:15:00 | 1103.60 | 1095.69 | 1094.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 1094.40 | 1098.10 | 1095.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 1094.40 | 1098.10 | 1095.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1094.40 | 1098.10 | 1095.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 1094.40 | 1098.10 | 1095.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1092.30 | 1096.94 | 1095.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:30:00 | 1092.40 | 1096.94 | 1095.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1093.00 | 1096.15 | 1095.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 1096.30 | 1096.08 | 1095.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 1098.40 | 1095.74 | 1095.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1160.90 | 1178.10 | 1178.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1160.90 | 1178.10 | 1178.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1157.80 | 1169.12 | 1174.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 11:15:00 | 1169.40 | 1164.41 | 1169.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 11:15:00 | 1169.40 | 1164.41 | 1169.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1169.40 | 1164.41 | 1169.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 1169.40 | 1164.41 | 1169.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1160.30 | 1163.59 | 1168.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 1159.90 | 1163.59 | 1168.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1159.00 | 1162.55 | 1167.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 1159.40 | 1161.92 | 1166.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1150.00 | 1161.58 | 1166.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1175.30 | 1162.07 | 1164.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 1175.30 | 1162.07 | 1164.64 | SL hit (close>static) qty=1.00 sl=1170.20 alert=retest2 |

### Cycle 105 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1182.20 | 1168.45 | 1167.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 1185.30 | 1176.44 | 1171.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 1176.00 | 1176.35 | 1172.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 1176.00 | 1176.35 | 1172.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1171.20 | 1175.41 | 1172.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 1171.20 | 1175.41 | 1172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1177.70 | 1175.87 | 1172.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 1177.20 | 1175.87 | 1172.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1183.60 | 1177.77 | 1174.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 1193.20 | 1181.28 | 1176.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 1193.20 | 1185.71 | 1180.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 1188.10 | 1188.67 | 1184.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1191.50 | 1188.54 | 1184.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1187.00 | 1188.23 | 1185.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 1181.90 | 1184.11 | 1184.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1181.90 | 1184.11 | 1184.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1174.80 | 1182.25 | 1183.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1185.10 | 1175.92 | 1175.82 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1166.20 | 1175.98 | 1177.01 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 1213.50 | 1181.07 | 1178.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1218.70 | 1197.96 | 1188.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 1215.40 | 1227.16 | 1218.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1215.40 | 1227.16 | 1218.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1215.40 | 1227.16 | 1218.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 1212.30 | 1227.16 | 1218.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1215.30 | 1224.79 | 1218.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 1217.40 | 1224.79 | 1218.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1212.20 | 1222.27 | 1217.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 1212.20 | 1222.27 | 1217.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1210.90 | 1220.00 | 1216.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 1211.00 | 1220.00 | 1216.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1215.10 | 1218.47 | 1216.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 1215.10 | 1218.47 | 1216.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1214.20 | 1217.62 | 1216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 1211.20 | 1217.62 | 1216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1218.10 | 1217.33 | 1216.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1214.10 | 1217.33 | 1216.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1202.50 | 1214.36 | 1215.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1196.60 | 1210.81 | 1213.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1199.20 | 1193.81 | 1200.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1199.20 | 1193.81 | 1200.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1199.20 | 1193.81 | 1200.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1199.20 | 1193.81 | 1200.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1198.10 | 1194.67 | 1200.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1230.40 | 1194.67 | 1200.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1223.60 | 1200.45 | 1202.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 1229.40 | 1200.45 | 1202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1235.30 | 1207.42 | 1205.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1238.00 | 1213.54 | 1208.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 1228.70 | 1229.01 | 1222.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 14:45:00 | 1228.40 | 1229.01 | 1222.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1223.40 | 1227.26 | 1222.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:30:00 | 1230.90 | 1227.14 | 1223.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:45:00 | 1229.80 | 1227.65 | 1224.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 1230.00 | 1231.59 | 1227.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 1227.00 | 1230.49 | 1227.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1231.40 | 1230.67 | 1228.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 1227.00 | 1230.67 | 1228.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1253.40 | 1244.61 | 1238.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:15:00 | 1261.90 | 1244.61 | 1238.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1256.80 | 1248.26 | 1241.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1226.60 | 1243.51 | 1244.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1226.60 | 1243.51 | 1244.51 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1247.00 | 1242.28 | 1241.74 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1237.80 | 1241.38 | 1241.38 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1245.60 | 1241.52 | 1241.35 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 1240.10 | 1241.24 | 1241.24 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1245.00 | 1241.99 | 1241.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1255.50 | 1244.61 | 1242.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1237.70 | 1248.15 | 1246.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1237.70 | 1248.15 | 1246.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1237.70 | 1248.15 | 1246.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 1237.70 | 1248.15 | 1246.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1236.00 | 1245.72 | 1245.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 1237.00 | 1245.72 | 1245.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1234.80 | 1243.54 | 1244.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1232.30 | 1240.39 | 1242.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1242.30 | 1236.33 | 1239.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 1242.30 | 1236.33 | 1239.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1242.30 | 1236.33 | 1239.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 1242.30 | 1236.33 | 1239.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1249.00 | 1238.86 | 1240.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 1249.00 | 1238.86 | 1240.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1247.50 | 1241.83 | 1241.67 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 1236.20 | 1240.70 | 1241.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 1232.00 | 1238.96 | 1240.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1244.90 | 1240.15 | 1240.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1244.90 | 1240.15 | 1240.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1244.90 | 1240.15 | 1240.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1245.00 | 1240.15 | 1240.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1240.00 | 1240.12 | 1240.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 1244.60 | 1240.12 | 1240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1233.00 | 1238.70 | 1239.98 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 1245.00 | 1240.14 | 1240.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 1247.80 | 1242.34 | 1241.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1264.90 | 1273.37 | 1267.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1264.90 | 1273.37 | 1267.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1264.90 | 1273.37 | 1267.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1264.90 | 1273.37 | 1267.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1268.00 | 1272.30 | 1267.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:00:00 | 1268.40 | 1271.52 | 1267.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 1271.00 | 1269.96 | 1267.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1260.70 | 1268.28 | 1267.37 | SL hit (close<static) qty=1.00 sl=1263.20 alert=retest2 |

### Cycle 122 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1256.40 | 1265.90 | 1266.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1251.60 | 1263.04 | 1265.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1266.70 | 1261.40 | 1263.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1266.70 | 1261.40 | 1263.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1266.70 | 1261.40 | 1263.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1267.70 | 1261.40 | 1263.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1265.70 | 1262.26 | 1263.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1217.00 | 1262.26 | 1263.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 1242.10 | 1239.95 | 1239.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1242.10 | 1239.95 | 1239.92 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1232.20 | 1238.63 | 1239.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 1230.00 | 1236.91 | 1238.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1203.70 | 1200.60 | 1213.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 12:15:00 | 1210.00 | 1204.13 | 1212.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1210.00 | 1204.13 | 1212.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 1210.00 | 1204.13 | 1212.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 1214.00 | 1206.10 | 1212.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 1214.00 | 1206.10 | 1212.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1216.00 | 1208.08 | 1212.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:00:00 | 1216.00 | 1208.08 | 1212.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1219.60 | 1210.38 | 1213.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1218.90 | 1210.38 | 1213.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1195.30 | 1207.37 | 1211.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 1192.10 | 1202.44 | 1208.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 14:00:00 | 1189.60 | 1199.87 | 1206.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 1132.49 | 1162.71 | 1178.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 1130.12 | 1148.63 | 1167.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 1141.50 | 1138.68 | 1155.69 | SL hit (close>ema200) qty=0.50 sl=1138.68 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1167.50 | 1155.03 | 1153.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1174.00 | 1161.26 | 1157.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1151.30 | 1169.05 | 1164.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1149.70 | 1165.18 | 1162.91 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1146.20 | 1161.38 | 1161.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1140.00 | 1157.11 | 1159.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1170.40 | 1150.49 | 1154.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1170.40 | 1150.49 | 1154.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1170.40 | 1150.49 | 1154.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1170.40 | 1150.49 | 1154.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1181.00 | 1156.59 | 1156.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1181.00 | 1156.59 | 1156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 1180.30 | 1161.33 | 1159.06 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1119.00 | 1157.45 | 1159.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1113.40 | 1148.64 | 1154.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1136.60 | 1124.73 | 1134.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1136.60 | 1124.73 | 1134.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1136.60 | 1124.73 | 1134.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1137.50 | 1124.73 | 1134.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1137.00 | 1127.18 | 1134.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1141.50 | 1127.18 | 1134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1136.50 | 1129.99 | 1134.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1168.40 | 1129.99 | 1134.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1163.80 | 1136.75 | 1137.20 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1159.70 | 1141.34 | 1139.24 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1128.80 | 1140.75 | 1142.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1125.20 | 1135.94 | 1139.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 1133.10 | 1132.39 | 1137.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:00:00 | 1133.10 | 1132.39 | 1137.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1154.40 | 1131.97 | 1134.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1154.40 | 1131.97 | 1134.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1147.90 | 1138.35 | 1137.31 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1114.00 | 1135.55 | 1136.94 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 1140.20 | 1133.33 | 1132.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 1151.10 | 1136.88 | 1134.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 12:15:00 | 1206.90 | 1209.18 | 1194.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 12:45:00 | 1208.70 | 1209.18 | 1194.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1193.50 | 1208.38 | 1198.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1196.50 | 1208.38 | 1198.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1206.60 | 1208.02 | 1199.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1209.80 | 1208.40 | 1200.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1223.70 | 1205.93 | 1201.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 1256.10 | 1262.55 | 1263.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1256.10 | 1262.55 | 1263.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1251.60 | 1258.85 | 1261.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 1257.20 | 1257.02 | 1259.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 1257.20 | 1257.02 | 1259.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1259.30 | 1257.48 | 1259.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 1260.30 | 1257.48 | 1259.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1253.30 | 1256.64 | 1259.17 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1289.90 | 1263.31 | 1261.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1301.30 | 1283.05 | 1274.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 1283.10 | 1285.82 | 1277.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:00:00 | 1283.10 | 1285.82 | 1277.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1277.60 | 1284.17 | 1277.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1278.70 | 1284.17 | 1277.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1280.30 | 1283.40 | 1278.16 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1253.20 | 1274.30 | 1276.44 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1273.00 | 1263.82 | 1263.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 1282.50 | 1270.42 | 1266.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1271.40 | 1277.10 | 1272.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1271.40 | 1277.10 | 1272.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1271.40 | 1277.10 | 1272.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 1271.00 | 1277.10 | 1272.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1278.00 | 1277.28 | 1273.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:30:00 | 1278.70 | 1277.59 | 1273.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:00:00 | 1278.80 | 1277.59 | 1273.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1284.00 | 1276.98 | 1274.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 11:15:00 | 705.50 | 2023-05-24 14:15:00 | 696.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-05-30 09:30:00 | 708.55 | 2023-05-31 09:15:00 | 697.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-05-30 15:15:00 | 705.30 | 2023-05-31 09:15:00 | 697.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-01 12:00:00 | 692.45 | 2023-06-02 14:15:00 | 706.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-06-01 15:15:00 | 692.95 | 2023-06-02 14:15:00 | 706.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2023-06-13 09:15:00 | 757.20 | 2023-06-21 09:15:00 | 760.25 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2023-07-20 11:00:00 | 788.65 | 2023-07-20 14:15:00 | 798.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-07-20 11:45:00 | 789.45 | 2023-07-20 14:15:00 | 798.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-08-04 10:45:00 | 805.75 | 2023-08-07 12:15:00 | 813.75 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-08-04 15:00:00 | 805.90 | 2023-08-07 12:15:00 | 813.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-08-07 09:45:00 | 806.00 | 2023-08-07 12:15:00 | 813.75 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-08-11 11:15:00 | 823.25 | 2023-08-14 09:15:00 | 802.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-08-11 11:45:00 | 823.05 | 2023-08-14 09:15:00 | 802.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2023-08-11 14:00:00 | 822.95 | 2023-08-14 09:15:00 | 802.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest1 | 2023-08-24 09:15:00 | 803.50 | 2023-08-24 11:15:00 | 795.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-08-29 14:45:00 | 780.40 | 2023-08-30 09:15:00 | 787.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-09-05 14:15:00 | 809.95 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-09-06 12:15:00 | 810.50 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-09-06 13:30:00 | 809.70 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2023-09-06 14:15:00 | 812.65 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-09-07 10:15:00 | 814.85 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-09-07 11:45:00 | 812.30 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-09-07 13:15:00 | 814.40 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-09-08 11:00:00 | 812.70 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-09-11 10:00:00 | 820.25 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-09-12 13:00:00 | 819.20 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2023-09-12 14:00:00 | 820.25 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-09-13 09:15:00 | 820.00 | 2023-09-13 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-09-18 11:30:00 | 809.70 | 2025-04-11 09:15:00 | 983.35 | STOP_HIT | 1.00 | -21.45% |
| SELL | retest2 | 2023-09-18 12:45:00 | 808.85 | 2025-04-11 09:15:00 | 983.35 | STOP_HIT | 1.00 | -21.57% |
| SELL | retest2 | 2023-09-18 14:45:00 | 809.60 | 2025-04-11 09:15:00 | 983.35 | STOP_HIT | 1.00 | -21.46% |
| SELL | retest2 | 2023-12-01 15:15:00 | 808.00 | 2025-04-11 09:15:00 | 983.35 | STOP_HIT | 1.00 | -21.70% |
| BUY | retest2 | 2025-04-24 12:30:00 | 1041.30 | 2025-04-25 10:15:00 | 1020.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-04-24 13:45:00 | 1042.50 | 2025-04-25 10:15:00 | 1020.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-02 11:15:00 | 988.40 | 2025-05-09 09:15:00 | 938.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 988.40 | 2025-05-09 11:15:00 | 958.30 | STOP_HIT | 0.50 | 3.05% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1010.50 | 2025-05-20 15:15:00 | 1013.70 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-05-29 10:30:00 | 1010.00 | 2025-06-06 10:15:00 | 1003.40 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1013.50 | 2025-06-12 12:15:00 | 999.85 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-11 09:45:00 | 1011.55 | 2025-06-12 12:15:00 | 999.85 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-19 12:30:00 | 990.45 | 2025-06-19 14:15:00 | 996.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-19 13:00:00 | 990.95 | 2025-06-19 14:15:00 | 996.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1047.20 | 2025-07-09 11:15:00 | 1038.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-07 12:45:00 | 1057.70 | 2025-08-08 12:15:00 | 1042.70 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-07 13:30:00 | 1057.00 | 2025-08-08 12:15:00 | 1042.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-08-07 14:00:00 | 1057.50 | 2025-08-08 12:15:00 | 1042.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-20 10:15:00 | 1077.60 | 2025-08-22 09:15:00 | 1057.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-08-21 13:45:00 | 1076.90 | 2025-08-22 09:15:00 | 1057.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1049.70 | 2025-09-02 11:15:00 | 1047.80 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-09-05 13:15:00 | 1070.00 | 2025-09-12 10:15:00 | 1093.60 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-09-18 10:15:00 | 1113.80 | 2025-09-26 13:15:00 | 1133.50 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2025-10-06 12:45:00 | 1155.90 | 2025-10-08 10:15:00 | 1148.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-08 10:15:00 | 1155.20 | 2025-10-08 10:15:00 | 1148.60 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-23 11:15:00 | 1143.00 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-23 13:45:00 | 1143.70 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-23 14:45:00 | 1143.70 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-24 10:15:00 | 1141.80 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-27 11:15:00 | 1145.00 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-27 11:45:00 | 1145.00 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-27 13:30:00 | 1145.00 | 2025-10-27 14:15:00 | 1152.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-11-17 11:15:00 | 1168.20 | 2025-11-20 14:15:00 | 1171.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-18 09:30:00 | 1167.60 | 2025-11-20 14:15:00 | 1171.30 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-11-20 11:30:00 | 1166.60 | 2025-11-20 14:15:00 | 1171.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-20 14:15:00 | 1170.00 | 2025-11-20 14:15:00 | 1171.30 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-11-27 14:15:00 | 1154.30 | 2025-12-03 10:15:00 | 1151.70 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-19 10:45:00 | 1080.50 | 2025-12-22 09:15:00 | 1092.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-19 11:30:00 | 1080.50 | 2025-12-22 09:15:00 | 1092.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-30 09:30:00 | 1096.30 | 2026-01-08 10:15:00 | 1160.90 | STOP_HIT | 1.00 | 5.89% |
| BUY | retest2 | 2025-12-30 13:15:00 | 1098.40 | 2026-01-08 10:15:00 | 1160.90 | STOP_HIT | 1.00 | 5.69% |
| SELL | retest2 | 2026-01-09 13:15:00 | 1159.90 | 2026-01-12 12:15:00 | 1175.30 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-09 13:45:00 | 1159.00 | 2026-01-12 12:15:00 | 1175.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-09 15:00:00 | 1159.40 | 2026-01-12 12:15:00 | 1175.30 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1150.00 | 2026-01-12 12:15:00 | 1175.30 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-01-14 10:45:00 | 1193.20 | 2026-01-20 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-01-16 10:00:00 | 1193.20 | 2026-01-20 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-01-16 14:45:00 | 1188.10 | 2026-01-20 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-01-19 09:15:00 | 1191.50 | 2026-01-20 09:15:00 | 1181.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-02-05 11:30:00 | 1230.90 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-02-05 12:45:00 | 1229.80 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-02-06 11:00:00 | 1230.00 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-02-06 13:15:00 | 1227.00 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-10 10:15:00 | 1261.90 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-02-10 11:30:00 | 1256.80 | 2026-02-13 09:15:00 | 1226.60 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2026-02-27 12:00:00 | 1268.40 | 2026-03-02 09:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-02-27 15:15:00 | 1271.00 | 2026-03-02 09:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1217.00 | 2026-03-06 12:15:00 | 1242.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1192.10 | 2026-03-13 10:15:00 | 1132.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:00:00 | 1189.60 | 2026-03-13 13:15:00 | 1130.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1192.10 | 2026-03-16 10:15:00 | 1141.50 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2026-03-11 14:00:00 | 1189.60 | 2026-03-16 10:15:00 | 1141.50 | STOP_HIT | 0.50 | 4.04% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1209.80 | 2026-04-23 13:15:00 | 1256.10 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1223.70 | 2026-04-23 13:15:00 | 1256.10 | STOP_HIT | 1.00 | 2.65% |
