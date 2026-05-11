# JSW Steel Ltd. (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1272.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 247 |
| ALERT1 | 161 |
| ALERT2 | 158 |
| ALERT2_SKIP | 89 |
| ALERT3 | 425 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 200 |
| PARTIAL | 17 |
| TARGET_HIT | 2 |
| STOP_HIT | 204 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 221 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 77 / 144
- **Target hits / Stop hits / Partials:** 2 / 203 / 16
- **Avg / median % per leg:** 0.28% / -0.65%
- **Sum % (uncompounded):** 62.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 119 | 42 | 35.3% | 0 | 116 | 3 | 0.01% | 1.1% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 5 | 3 | 3.27% | 26.2% |
| BUY @ 3rd Alert (retest2) | 111 | 36 | 32.4% | 0 | 111 | 0 | -0.23% | -25.1% |
| SELL (all) | 102 | 35 | 34.3% | 2 | 87 | 13 | 0.60% | 61.3% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.25% | 0.8% |
| SELL @ 3rd Alert (retest2) | 99 | 32 | 32.3% | 2 | 84 | 13 | 0.61% | 60.5% |
| retest1 (combined) | 11 | 9 | 81.8% | 0 | 8 | 3 | 2.45% | 26.9% |
| retest2 (combined) | 210 | 68 | 32.4% | 2 | 195 | 13 | 0.17% | 35.4% |

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
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 13:15:00 | 769.22 | 777.38 | 779.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 13:15:00 | 768.41 | 777.38 | 779.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 13:15:00 | 769.12 | 777.38 | 779.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-29 09:15:00 | 774.90 | 774.23 | 777.13 | SL hit (close>ema200) qty=0.50 sl=774.23 alert=retest2 |

### Cycle 29 — BUY (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 14:15:00 | 779.50 | 778.12 | 778.11 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 769.30 | 776.49 | 777.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 760.30 | 769.45 | 772.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 15:15:00 | 757.10 | 756.51 | 760.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:15:00 | 764.20 | 756.51 | 760.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 760.50 | 757.31 | 760.80 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 763.00 | 762.13 | 762.09 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 755.55 | 760.82 | 761.49 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 764.50 | 760.43 | 760.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 773.15 | 762.98 | 761.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 774.60 | 780.90 | 776.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 774.60 | 780.90 | 776.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 774.60 | 780.90 | 776.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:00:00 | 774.60 | 780.90 | 776.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 775.55 | 779.83 | 776.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:30:00 | 773.10 | 779.83 | 776.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 774.50 | 778.77 | 776.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 11:30:00 | 774.35 | 778.77 | 776.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 777.95 | 778.42 | 776.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:45:00 | 777.30 | 778.42 | 776.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 777.65 | 778.27 | 776.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:45:00 | 777.35 | 778.27 | 776.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 775.00 | 777.62 | 776.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 779.30 | 777.62 | 776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 785.10 | 779.11 | 777.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 11:30:00 | 787.00 | 782.15 | 779.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 09:15:00 | 775.90 | 785.26 | 786.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 775.90 | 785.26 | 786.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 768.00 | 775.43 | 779.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 15:15:00 | 771.65 | 770.47 | 775.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 15:15:00 | 771.65 | 770.47 | 775.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 771.65 | 770.47 | 775.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 12:00:00 | 761.85 | 767.47 | 772.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:00:00 | 761.65 | 766.30 | 771.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:30:00 | 760.80 | 765.63 | 771.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 14:15:00 | 761.40 | 765.63 | 771.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 734.25 | 733.62 | 736.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 14:45:00 | 733.35 | 733.62 | 736.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 730.90 | 733.21 | 735.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 10:30:00 | 727.25 | 732.65 | 735.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 09:15:00 | 727.50 | 734.02 | 735.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 09:15:00 | 723.76 | 732.59 | 734.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 09:15:00 | 723.57 | 732.59 | 734.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 09:15:00 | 723.33 | 732.59 | 734.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 12:00:00 | 729.45 | 731.46 | 733.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 09:15:00 | 728.75 | 727.94 | 730.70 | SL hit (close>ema200) qty=0.50 sl=727.94 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 741.05 | 732.90 | 731.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 10:15:00 | 744.85 | 735.29 | 733.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 751.30 | 752.45 | 747.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 13:00:00 | 751.30 | 752.45 | 747.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 748.75 | 750.93 | 748.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:15:00 | 749.15 | 750.93 | 748.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 749.70 | 750.68 | 748.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 13:00:00 | 755.10 | 750.77 | 748.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:45:00 | 753.30 | 752.55 | 750.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 13:00:00 | 753.60 | 752.51 | 750.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 14:00:00 | 754.40 | 752.89 | 751.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 750.85 | 752.48 | 751.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 750.85 | 752.48 | 751.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 750.45 | 752.08 | 751.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 748.80 | 752.08 | 751.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 751.80 | 752.02 | 751.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:15:00 | 752.00 | 752.02 | 751.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 13:15:00 | 763.95 | 766.92 | 767.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 763.95 | 766.92 | 767.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 14:15:00 | 761.15 | 765.76 | 766.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 772.05 | 766.58 | 766.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 772.05 | 766.58 | 766.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 772.05 | 766.58 | 766.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 774.20 | 766.58 | 766.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 10:15:00 | 779.20 | 769.10 | 768.04 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 14:15:00 | 768.00 | 770.29 | 770.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 771.60 | 770.65 | 770.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 11:15:00 | 773.15 | 771.15 | 770.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 15:15:00 | 779.85 | 780.36 | 777.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:15:00 | 783.75 | 780.36 | 777.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 15:00:00 | 782.70 | 781.25 | 779.04 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:15:00 | 784.65 | 781.42 | 779.32 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 11:15:00 | 821.84 | 811.91 | 803.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 12:15:00 | 822.94 | 813.56 | 805.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 10:15:00 | 823.88 | 818.60 | 811.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 816.85 | 818.25 | 811.70 | SL hit (close<ema200) qty=0.50 sl=818.25 alert=retest1 |

### Cycle 40 — SELL (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 09:15:00 | 848.30 | 854.48 | 854.82 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 13:15:00 | 859.35 | 854.68 | 854.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 10:15:00 | 860.25 | 857.47 | 856.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 852.80 | 856.77 | 856.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 852.80 | 856.77 | 856.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 852.80 | 856.77 | 856.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 852.80 | 856.77 | 856.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 844.35 | 854.29 | 855.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 836.10 | 847.01 | 851.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 843.75 | 843.61 | 847.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 845.70 | 843.61 | 847.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 848.35 | 844.56 | 847.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:45:00 | 848.35 | 844.56 | 847.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 847.25 | 845.10 | 847.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:15:00 | 845.20 | 845.86 | 847.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 13:15:00 | 850.70 | 846.83 | 847.62 | SL hit (close>static) qty=1.00 sl=848.90 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 856.00 | 848.66 | 848.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 861.10 | 852.24 | 850.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 13:15:00 | 854.70 | 855.32 | 852.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 13:45:00 | 854.65 | 855.32 | 852.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 851.85 | 854.63 | 852.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 15:00:00 | 851.85 | 854.63 | 852.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 854.90 | 854.68 | 852.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:15:00 | 859.10 | 854.68 | 852.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 863.55 | 874.90 | 875.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 863.55 | 874.90 | 875.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 846.70 | 866.03 | 870.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 843.50 | 840.32 | 847.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 10:00:00 | 843.50 | 840.32 | 847.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 825.70 | 823.80 | 826.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 14:00:00 | 825.70 | 823.80 | 826.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 829.70 | 824.98 | 826.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 829.70 | 824.98 | 826.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 15:15:00 | 830.25 | 826.03 | 827.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:15:00 | 833.55 | 826.03 | 827.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 821.35 | 825.21 | 826.46 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 827.80 | 825.46 | 825.38 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 822.50 | 825.59 | 825.65 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 13:15:00 | 837.60 | 827.99 | 826.74 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 816.60 | 825.44 | 826.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 813.60 | 820.37 | 823.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 814.90 | 807.82 | 812.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 814.90 | 807.82 | 812.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 814.90 | 807.82 | 812.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 814.90 | 807.82 | 812.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 813.80 | 809.01 | 812.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:15:00 | 816.40 | 809.01 | 812.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 814.75 | 810.16 | 813.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 12:15:00 | 812.50 | 810.16 | 813.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 13:15:00 | 813.30 | 811.26 | 813.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 15:15:00 | 818.25 | 814.10 | 814.20 | SL hit (close>static) qty=1.00 sl=817.95 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 811.40 | 806.22 | 805.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 813.90 | 809.49 | 807.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 09:15:00 | 811.20 | 811.27 | 808.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 811.20 | 811.27 | 808.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 811.20 | 811.27 | 808.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:45:00 | 811.70 | 811.27 | 808.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 816.45 | 812.31 | 809.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 09:15:00 | 819.90 | 812.36 | 810.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:30:00 | 820.90 | 816.82 | 815.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:00:00 | 819.25 | 816.38 | 815.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 12:15:00 | 808.70 | 813.72 | 814.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 12:15:00 | 808.70 | 813.72 | 814.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 13:15:00 | 805.30 | 812.03 | 813.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 808.80 | 808.57 | 811.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 808.80 | 808.57 | 811.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 808.80 | 808.57 | 811.34 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 13:15:00 | 820.15 | 814.13 | 813.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 14:15:00 | 822.55 | 815.81 | 814.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 815.75 | 819.99 | 817.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 14:15:00 | 815.75 | 819.99 | 817.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 815.75 | 819.99 | 817.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 813.35 | 819.99 | 817.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 812.85 | 818.56 | 817.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 807.00 | 818.56 | 817.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 10:15:00 | 811.35 | 815.89 | 816.33 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 828.15 | 818.37 | 817.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 12:15:00 | 839.90 | 826.10 | 821.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 824.70 | 831.32 | 826.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 824.70 | 831.32 | 826.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 824.70 | 831.32 | 826.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 824.70 | 831.32 | 826.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 827.25 | 830.51 | 826.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:30:00 | 823.75 | 830.51 | 826.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 828.00 | 830.01 | 826.54 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 814.25 | 823.73 | 824.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 805.50 | 820.08 | 822.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 815.40 | 810.96 | 814.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 13:15:00 | 815.40 | 810.96 | 814.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 815.40 | 810.96 | 814.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:00:00 | 815.40 | 810.96 | 814.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 810.40 | 810.85 | 814.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 15:15:00 | 808.25 | 810.85 | 814.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:15:00 | 807.30 | 809.62 | 812.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 12:00:00 | 808.10 | 809.32 | 812.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 800.20 | 810.66 | 812.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 812.60 | 807.97 | 809.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 812.60 | 807.97 | 809.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 813.40 | 809.05 | 809.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 814.80 | 809.05 | 809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-15 10:15:00 | 813.00 | 810.91 | 810.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 813.00 | 810.91 | 810.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 817.40 | 812.70 | 811.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 816.90 | 817.86 | 815.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 816.90 | 817.86 | 815.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 816.90 | 817.86 | 815.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 814.20 | 817.86 | 815.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 815.50 | 819.28 | 817.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 815.50 | 819.28 | 817.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 813.65 | 818.15 | 817.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 813.65 | 818.15 | 817.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 817.80 | 818.08 | 817.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:45:00 | 819.25 | 818.70 | 817.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:00:00 | 821.45 | 818.95 | 818.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 11:15:00 | 823.35 | 827.09 | 827.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 11:15:00 | 823.35 | 827.09 | 827.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 821.40 | 825.38 | 826.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 10:15:00 | 815.65 | 812.90 | 817.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 10:45:00 | 815.25 | 812.90 | 817.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 818.80 | 814.56 | 817.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:45:00 | 818.70 | 814.56 | 817.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 817.15 | 815.08 | 817.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:30:00 | 817.00 | 815.08 | 817.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 819.45 | 815.95 | 817.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 819.45 | 815.95 | 817.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 818.50 | 816.46 | 817.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 818.05 | 816.46 | 817.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 815.50 | 816.27 | 817.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:15:00 | 812.55 | 816.27 | 817.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 830.50 | 807.77 | 808.36 | SL hit (close>static) qty=1.00 sl=822.75 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 826.50 | 811.52 | 810.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 832.80 | 815.77 | 812.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 822.55 | 831.78 | 824.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 822.55 | 831.78 | 824.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 822.55 | 831.78 | 824.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 822.55 | 831.78 | 824.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 824.55 | 830.33 | 824.40 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 812.25 | 822.98 | 823.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 806.30 | 817.18 | 819.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 813.30 | 810.43 | 814.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 15:00:00 | 813.30 | 810.43 | 814.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 836.70 | 816.10 | 816.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:45:00 | 838.90 | 816.10 | 816.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 832.85 | 819.45 | 818.17 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 820.40 | 823.97 | 824.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 14:15:00 | 816.35 | 821.75 | 823.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 12:15:00 | 785.90 | 784.64 | 792.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-15 13:00:00 | 785.90 | 784.64 | 792.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 795.65 | 786.16 | 790.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:00:00 | 795.65 | 786.16 | 790.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 794.65 | 787.86 | 791.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:30:00 | 795.00 | 787.86 | 791.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 12:15:00 | 805.80 | 793.61 | 793.25 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 795.80 | 796.99 | 797.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 793.15 | 796.23 | 796.68 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 812.45 | 799.47 | 798.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 821.55 | 813.12 | 807.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 10:15:00 | 819.70 | 820.05 | 814.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 11:00:00 | 819.70 | 820.05 | 814.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 822.30 | 821.80 | 817.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 13:00:00 | 827.75 | 823.10 | 819.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 14:30:00 | 825.35 | 822.46 | 819.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:45:00 | 826.30 | 823.08 | 820.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 09:15:00 | 851.10 | 861.51 | 862.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 09:15:00 | 851.10 | 861.51 | 862.83 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 869.50 | 862.87 | 862.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 13:15:00 | 872.95 | 865.99 | 863.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 12:15:00 | 877.00 | 877.14 | 871.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 13:00:00 | 877.00 | 877.14 | 871.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 873.00 | 875.95 | 872.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 878.10 | 875.95 | 872.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 869.00 | 878.75 | 876.45 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 866.75 | 874.18 | 874.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 865.30 | 872.40 | 873.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 10:15:00 | 854.75 | 851.12 | 857.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 11:00:00 | 854.75 | 851.12 | 857.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 855.05 | 851.90 | 857.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:00:00 | 855.05 | 851.90 | 857.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 857.55 | 853.03 | 857.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:00:00 | 857.55 | 853.03 | 857.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 849.00 | 852.23 | 856.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:45:00 | 845.25 | 850.62 | 855.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 842.10 | 850.30 | 854.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 14:15:00 | 866.20 | 852.81 | 853.58 | SL hit (close>static) qty=1.00 sl=858.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 864.00 | 855.04 | 854.53 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 850.95 | 855.81 | 856.44 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 865.20 | 857.69 | 857.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 10:15:00 | 871.55 | 860.46 | 858.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 891.70 | 894.40 | 886.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 13:45:00 | 893.35 | 894.40 | 886.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 888.70 | 893.26 | 886.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 888.70 | 893.26 | 886.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 890.00 | 892.61 | 886.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:30:00 | 891.45 | 893.58 | 887.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 10:30:00 | 892.00 | 892.78 | 890.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:15:00 | 891.80 | 892.32 | 890.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:45:00 | 892.05 | 892.20 | 890.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 890.75 | 891.91 | 890.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:00:00 | 890.75 | 891.91 | 890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 882.10 | 889.95 | 889.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 882.10 | 889.95 | 889.89 | SL hit (close<static) qty=1.00 sl=886.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 883.90 | 888.74 | 889.35 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 893.20 | 889.81 | 889.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 896.50 | 891.36 | 890.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 10:15:00 | 888.85 | 890.86 | 890.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 10:15:00 | 888.85 | 890.86 | 890.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 888.85 | 890.86 | 890.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:00:00 | 888.85 | 890.86 | 890.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 874.60 | 887.61 | 888.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 12:15:00 | 871.40 | 884.37 | 887.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 887.95 | 881.02 | 883.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 887.95 | 881.02 | 883.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 887.95 | 881.02 | 883.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 887.95 | 881.02 | 883.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 891.05 | 883.03 | 884.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:45:00 | 892.90 | 883.03 | 884.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 13:15:00 | 885.70 | 883.77 | 884.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 13:45:00 | 886.70 | 883.77 | 884.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 881.35 | 883.28 | 884.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 15:15:00 | 880.00 | 883.28 | 884.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 14:15:00 | 836.00 | 848.11 | 857.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 852.60 | 846.75 | 854.85 | SL hit (close>ema200) qty=0.50 sl=846.75 alert=retest2 |

### Cycle 73 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 862.00 | 854.02 | 853.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 874.10 | 858.04 | 855.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 868.60 | 873.05 | 865.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 868.60 | 873.05 | 865.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 874.45 | 873.33 | 866.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 865.55 | 873.33 | 866.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 870.60 | 872.22 | 869.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 870.50 | 872.22 | 869.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 869.35 | 871.64 | 869.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:45:00 | 867.15 | 871.64 | 869.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 875.60 | 872.43 | 869.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:15:00 | 876.80 | 872.91 | 870.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 905.75 | 910.94 | 911.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 905.75 | 910.94 | 911.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 901.60 | 909.05 | 910.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 908.15 | 906.37 | 908.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 908.15 | 906.37 | 908.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 908.15 | 906.37 | 908.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 907.60 | 906.37 | 908.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 909.75 | 907.05 | 908.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 14:45:00 | 907.95 | 907.08 | 908.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 907.35 | 905.88 | 907.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 912.35 | 896.28 | 894.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 912.35 | 896.28 | 894.93 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 860.20 | 896.95 | 897.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 834.00 | 884.36 | 892.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 857.95 | 857.56 | 870.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:15:00 | 857.20 | 857.56 | 870.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 869.70 | 859.99 | 870.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 869.70 | 859.99 | 870.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 879.90 | 863.97 | 871.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 879.90 | 863.97 | 871.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 875.00 | 866.17 | 871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 887.00 | 866.17 | 871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 886.15 | 870.17 | 872.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 887.00 | 870.17 | 872.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 885.60 | 876.67 | 875.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 895.00 | 884.67 | 880.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 12:15:00 | 912.15 | 912.43 | 905.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:30:00 | 913.10 | 912.43 | 905.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 913.80 | 916.03 | 912.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 914.70 | 916.03 | 912.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 908.10 | 914.44 | 912.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 908.10 | 914.44 | 912.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 914.85 | 914.52 | 912.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:30:00 | 917.85 | 915.02 | 912.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:45:00 | 917.05 | 915.54 | 913.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:00:00 | 916.65 | 915.13 | 913.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:30:00 | 917.80 | 920.93 | 920.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 931.45 | 923.33 | 921.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:30:00 | 924.00 | 923.33 | 921.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 927.70 | 929.90 | 926.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 927.70 | 929.90 | 926.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 942.50 | 932.42 | 928.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-25 15:15:00 | 929.80 | 931.18 | 931.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 929.80 | 931.18 | 931.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 925.25 | 929.99 | 930.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 935.60 | 926.05 | 927.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 935.60 | 926.05 | 927.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 935.60 | 926.05 | 927.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 937.20 | 926.05 | 927.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 936.20 | 928.08 | 928.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 933.50 | 928.08 | 928.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 942.45 | 930.96 | 929.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 943.30 | 934.97 | 931.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 12:15:00 | 938.00 | 938.13 | 934.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 12:15:00 | 938.00 | 938.13 | 934.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 938.00 | 938.13 | 934.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 935.70 | 938.13 | 934.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 937.25 | 937.95 | 935.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:30:00 | 935.15 | 937.95 | 935.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 929.65 | 936.29 | 934.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 929.65 | 936.29 | 934.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 934.10 | 935.85 | 934.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 940.90 | 935.85 | 934.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 939.10 | 946.89 | 947.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 939.10 | 946.89 | 947.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 12:15:00 | 937.35 | 944.98 | 946.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 942.85 | 942.20 | 944.62 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:15:00 | 934.95 | 942.20 | 944.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:45:00 | 935.60 | 939.12 | 942.45 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 13:45:00 | 935.55 | 937.64 | 941.15 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 923.50 | 925.66 | 931.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 921.50 | 925.66 | 931.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 920.35 | 923.94 | 929.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:30:00 | 922.15 | 923.25 | 927.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 933.00 | 925.86 | 927.77 | SL hit (close>ema400) qty=1.00 sl=927.77 alert=retest1 |

### Cycle 81 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 935.10 | 929.45 | 929.07 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 925.65 | 931.72 | 931.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 912.70 | 926.70 | 929.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 888.25 | 887.06 | 895.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 14:00:00 | 888.25 | 887.06 | 895.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 893.05 | 888.26 | 895.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 893.05 | 888.26 | 895.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 886.10 | 888.74 | 894.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:30:00 | 881.30 | 887.40 | 892.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 882.55 | 886.49 | 891.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 881.25 | 885.45 | 890.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 895.45 | 880.46 | 883.44 | SL hit (close>static) qty=1.00 sl=895.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 899.45 | 887.17 | 886.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 902.80 | 890.30 | 887.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 898.95 | 899.34 | 894.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 898.95 | 899.34 | 894.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 900.95 | 899.20 | 896.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 903.50 | 899.20 | 896.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:15:00 | 903.90 | 900.60 | 897.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 903.15 | 901.25 | 898.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 903.00 | 901.41 | 898.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 909.15 | 925.59 | 920.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 909.15 | 925.59 | 920.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 905.00 | 921.47 | 918.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 905.00 | 921.47 | 918.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-02 12:15:00 | 908.85 | 916.32 | 916.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 908.85 | 916.32 | 916.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 899.60 | 911.32 | 914.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 888.25 | 876.19 | 889.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 888.25 | 876.19 | 889.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 888.25 | 876.19 | 889.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 888.25 | 876.19 | 889.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 883.00 | 877.55 | 888.60 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 908.55 | 893.61 | 891.69 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 887.00 | 894.35 | 894.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 883.85 | 892.25 | 893.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 894.85 | 892.77 | 893.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 894.85 | 892.77 | 893.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 894.85 | 892.77 | 893.73 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 910.00 | 897.05 | 895.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 917.85 | 906.68 | 901.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 916.60 | 917.78 | 912.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:00:00 | 916.60 | 917.78 | 912.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 913.75 | 916.98 | 912.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 913.75 | 916.98 | 912.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 909.00 | 915.38 | 911.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 909.00 | 915.38 | 911.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 907.50 | 913.81 | 911.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 908.20 | 913.81 | 911.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 893.40 | 907.53 | 909.02 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 913.25 | 902.88 | 902.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 924.55 | 916.88 | 912.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 937.95 | 938.55 | 933.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 11:00:00 | 937.95 | 938.55 | 933.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 944.65 | 949.65 | 945.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 943.10 | 949.65 | 945.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 943.80 | 948.48 | 945.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 940.35 | 948.48 | 945.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 937.70 | 946.32 | 944.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 937.25 | 946.32 | 944.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 936.90 | 944.44 | 943.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 936.90 | 944.44 | 943.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 950.30 | 946.81 | 945.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 946.95 | 946.81 | 945.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 949.00 | 947.58 | 945.80 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 939.00 | 944.71 | 944.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 935.50 | 942.90 | 943.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 946.20 | 942.45 | 943.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 946.20 | 942.45 | 943.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 946.20 | 942.45 | 943.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 943.10 | 942.45 | 943.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 951.20 | 944.20 | 944.07 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 940.15 | 943.76 | 944.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 09:15:00 | 935.80 | 941.16 | 942.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 13:15:00 | 931.70 | 928.59 | 933.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 14:00:00 | 931.70 | 928.59 | 933.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 933.65 | 929.60 | 933.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 933.65 | 929.60 | 933.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 932.45 | 930.17 | 933.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 935.75 | 930.17 | 933.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 941.75 | 932.49 | 933.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 941.95 | 932.49 | 933.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 937.25 | 933.44 | 934.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 935.35 | 933.44 | 934.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:30:00 | 933.75 | 927.00 | 929.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 934.05 | 931.01 | 930.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 934.05 | 931.01 | 930.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 936.40 | 932.59 | 931.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 933.60 | 936.81 | 934.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 933.60 | 936.81 | 934.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 933.60 | 936.81 | 934.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 933.60 | 936.81 | 934.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 936.95 | 936.83 | 934.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 12:00:00 | 938.80 | 937.23 | 935.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 13:15:00 | 928.15 | 935.09 | 934.52 | SL hit (close<static) qty=1.00 sl=932.45 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 927.00 | 933.47 | 933.84 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 948.35 | 935.70 | 934.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 955.45 | 939.65 | 936.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 953.00 | 953.72 | 947.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:15:00 | 960.70 | 953.72 | 947.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 967.70 | 965.71 | 961.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 956.45 | 963.80 | 962.01 | SL hit (close<ema400) qty=1.00 sl=962.01 alert=retest1 |

### Cycle 96 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 957.00 | 960.33 | 960.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 947.30 | 954.62 | 957.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 978.75 | 957.06 | 957.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 978.75 | 957.06 | 957.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 978.75 | 957.06 | 957.34 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 984.35 | 962.52 | 959.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1002.20 | 984.89 | 977.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 985.75 | 988.34 | 982.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 984.00 | 987.47 | 982.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 984.00 | 987.47 | 982.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 984.00 | 987.47 | 982.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 979.00 | 985.78 | 982.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 979.45 | 985.78 | 982.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 974.35 | 983.49 | 981.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 974.35 | 983.49 | 981.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 983.00 | 981.93 | 981.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:45:00 | 984.00 | 983.01 | 981.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1020.00 | 1031.87 | 1033.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 1020.00 | 1031.87 | 1033.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 988.25 | 1017.02 | 1025.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 1006.65 | 1002.16 | 1010.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 10:45:00 | 1006.10 | 1002.16 | 1010.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1000.95 | 1001.92 | 1009.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 998.90 | 1001.69 | 1008.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 998.10 | 1000.03 | 1004.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:00:00 | 999.15 | 1000.03 | 1004.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 15:15:00 | 1012.20 | 1004.47 | 1005.54 | SL hit (close>static) qty=1.00 sl=1011.50 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 1024.25 | 1008.42 | 1007.24 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 1001.10 | 1011.18 | 1011.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 998.50 | 1007.15 | 1009.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 982.00 | 981.39 | 989.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 15:00:00 | 982.00 | 981.39 | 989.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 987.35 | 982.99 | 988.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 987.00 | 982.99 | 988.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 994.75 | 985.34 | 988.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 994.75 | 985.34 | 988.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 992.35 | 986.75 | 989.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 995.55 | 986.75 | 989.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 995.95 | 990.50 | 990.42 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 986.50 | 989.70 | 990.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 981.25 | 987.35 | 988.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 12:15:00 | 960.25 | 959.70 | 965.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 12:45:00 | 958.95 | 959.70 | 965.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 950.20 | 946.86 | 953.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 940.35 | 946.86 | 953.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 955.85 | 948.66 | 953.99 | SL hit (close>static) qty=1.00 sl=954.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 968.85 | 958.80 | 957.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 15:15:00 | 972.15 | 964.57 | 961.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 958.95 | 963.45 | 960.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 958.95 | 963.45 | 960.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 958.95 | 963.45 | 960.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 958.95 | 963.45 | 960.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 947.75 | 960.31 | 959.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 947.75 | 960.31 | 959.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 953.75 | 959.00 | 959.09 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 966.50 | 960.19 | 959.33 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 955.60 | 959.44 | 959.63 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 963.65 | 960.28 | 959.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 970.60 | 962.88 | 961.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 954.70 | 962.06 | 961.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 954.70 | 962.06 | 961.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 954.70 | 962.06 | 961.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 954.70 | 962.06 | 961.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 944.20 | 958.49 | 959.67 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 981.30 | 962.59 | 960.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 12:15:00 | 985.80 | 970.00 | 964.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 998.30 | 1001.40 | 990.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:00:00 | 998.30 | 1001.40 | 990.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 989.85 | 997.67 | 990.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 989.85 | 997.67 | 990.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 992.70 | 996.67 | 990.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 986.70 | 996.67 | 990.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 989.60 | 995.26 | 990.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 989.60 | 995.26 | 990.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 990.05 | 994.22 | 990.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 997.45 | 994.22 | 990.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 991.65 | 993.70 | 990.80 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 983.05 | 988.54 | 989.04 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 996.75 | 990.18 | 989.74 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 985.35 | 989.25 | 989.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 972.30 | 985.28 | 987.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 946.45 | 942.82 | 955.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 946.45 | 942.82 | 955.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 946.60 | 941.68 | 946.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 946.60 | 941.68 | 946.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 952.00 | 943.74 | 947.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 951.95 | 943.74 | 947.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 954.75 | 945.94 | 948.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 954.75 | 945.94 | 948.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 950.00 | 947.50 | 948.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 953.50 | 947.50 | 948.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 954.65 | 949.97 | 949.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 959.90 | 952.67 | 950.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 947.25 | 951.64 | 950.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 947.25 | 951.64 | 950.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 947.25 | 951.64 | 950.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 947.25 | 951.64 | 950.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 945.00 | 950.31 | 950.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 944.35 | 950.31 | 950.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 938.75 | 948.00 | 949.15 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 951.75 | 948.35 | 948.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 955.85 | 949.85 | 948.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 09:15:00 | 955.40 | 962.31 | 956.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 955.40 | 962.31 | 956.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 955.40 | 962.31 | 956.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 955.40 | 962.31 | 956.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 961.10 | 962.07 | 957.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 14:30:00 | 964.65 | 959.62 | 957.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:45:00 | 965.65 | 961.57 | 958.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 963.15 | 962.21 | 959.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 953.55 | 961.94 | 962.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 953.55 | 961.94 | 962.02 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 968.75 | 962.37 | 962.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 975.25 | 967.06 | 964.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 987.00 | 992.86 | 986.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 987.00 | 992.86 | 986.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 987.00 | 992.86 | 986.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 987.00 | 992.86 | 986.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 989.65 | 992.22 | 986.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:00:00 | 991.85 | 992.14 | 987.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:30:00 | 996.10 | 992.80 | 988.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 980.95 | 990.65 | 987.94 | SL hit (close<static) qty=1.00 sl=985.50 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 998.00 | 1006.40 | 1006.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 994.20 | 1003.96 | 1005.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 1007.95 | 1003.06 | 1004.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 12:15:00 | 1007.95 | 1003.06 | 1004.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1007.95 | 1003.06 | 1004.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:45:00 | 1011.40 | 1003.06 | 1004.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 1006.40 | 1003.73 | 1004.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:45:00 | 1008.00 | 1003.73 | 1004.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1006.95 | 1004.37 | 1005.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 1006.95 | 1004.37 | 1005.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1002.40 | 1003.98 | 1004.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 982.70 | 1003.98 | 1004.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 933.57 | 950.01 | 963.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 934.10 | 930.00 | 943.55 | SL hit (close>ema200) qty=0.50 sl=930.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 914.25 | 907.76 | 907.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 919.80 | 911.26 | 908.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 915.65 | 915.67 | 912.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 915.65 | 915.67 | 912.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 913.00 | 915.14 | 912.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 910.10 | 915.14 | 912.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 908.20 | 913.75 | 912.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 907.15 | 913.75 | 912.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 903.40 | 911.68 | 911.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 903.40 | 911.68 | 911.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 905.40 | 910.42 | 910.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 899.95 | 906.64 | 908.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 908.80 | 906.37 | 908.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 908.80 | 906.37 | 908.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 908.80 | 906.37 | 908.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 911.50 | 906.37 | 908.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 907.30 | 906.55 | 908.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 909.40 | 906.55 | 908.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 908.20 | 906.88 | 908.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 910.20 | 906.88 | 908.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 908.60 | 907.23 | 908.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 908.95 | 907.23 | 908.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 911.15 | 908.01 | 908.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 910.45 | 908.01 | 908.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 912.35 | 908.88 | 908.83 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 907.05 | 908.80 | 908.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 902.05 | 907.45 | 908.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 904.85 | 904.05 | 906.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 904.85 | 904.05 | 906.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 902.75 | 903.79 | 905.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 900.00 | 903.79 | 905.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 897.70 | 902.69 | 904.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:30:00 | 900.10 | 901.06 | 903.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:00:00 | 900.25 | 901.06 | 903.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 898.15 | 895.35 | 898.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:30:00 | 895.25 | 895.35 | 898.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 900.90 | 896.46 | 899.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:15:00 | 900.00 | 896.46 | 899.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 896.85 | 896.54 | 898.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 891.90 | 896.97 | 898.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 10:45:00 | 894.25 | 895.65 | 897.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 895.75 | 891.25 | 894.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 905.50 | 894.10 | 895.13 | SL hit (close>static) qty=1.00 sl=903.40 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 903.50 | 895.98 | 895.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 909.25 | 902.81 | 899.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 903.35 | 905.12 | 902.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 14:00:00 | 903.35 | 905.12 | 902.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 903.40 | 904.78 | 902.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 905.25 | 904.78 | 902.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 900.25 | 905.17 | 903.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:00:00 | 900.25 | 905.17 | 903.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 902.50 | 904.63 | 903.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:00:00 | 907.95 | 904.95 | 903.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 15:15:00 | 909.70 | 905.27 | 903.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:00:00 | 908.35 | 907.90 | 905.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:00:00 | 909.75 | 909.10 | 907.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 919.50 | 911.18 | 908.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 922.00 | 911.18 | 908.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:00:00 | 921.05 | 914.24 | 910.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 922.00 | 916.00 | 911.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 15:00:00 | 920.75 | 919.40 | 918.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 917.60 | 919.04 | 918.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 916.90 | 919.04 | 918.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 922.30 | 919.69 | 919.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 929.60 | 919.69 | 919.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 911.40 | 927.46 | 927.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 911.40 | 927.46 | 927.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 907.15 | 917.68 | 922.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 917.30 | 916.66 | 920.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 917.30 | 916.66 | 920.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 917.05 | 915.85 | 919.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:45:00 | 915.15 | 915.85 | 919.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 923.00 | 916.91 | 919.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 923.00 | 916.91 | 919.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 931.55 | 919.84 | 920.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 931.55 | 919.84 | 920.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 939.05 | 923.68 | 922.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 954.30 | 936.20 | 929.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 942.45 | 947.44 | 942.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 942.45 | 947.44 | 942.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 942.45 | 947.44 | 942.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 942.45 | 947.44 | 942.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 945.20 | 946.99 | 942.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:30:00 | 945.75 | 946.99 | 942.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 947.45 | 946.68 | 943.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 951.00 | 947.65 | 943.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:00:00 | 951.50 | 947.65 | 943.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 919.00 | 941.28 | 941.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 919.00 | 941.28 | 941.60 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 945.10 | 937.39 | 936.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 949.00 | 941.81 | 939.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 945.65 | 949.32 | 946.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 12:15:00 | 945.65 | 949.32 | 946.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 945.65 | 949.32 | 946.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 945.65 | 949.32 | 946.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 943.55 | 948.16 | 945.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:30:00 | 942.50 | 948.16 | 945.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 948.15 | 948.16 | 946.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 960.45 | 947.98 | 946.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 954.70 | 964.24 | 959.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 13:15:00 | 953.45 | 958.46 | 958.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 13:15:00 | 953.45 | 958.46 | 958.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 943.20 | 953.80 | 956.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 959.20 | 953.42 | 955.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 959.20 | 953.42 | 955.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 959.20 | 953.42 | 955.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 959.20 | 953.42 | 955.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 957.20 | 954.18 | 955.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:30:00 | 956.50 | 954.18 | 955.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 952.85 | 953.91 | 955.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 956.60 | 953.91 | 955.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 957.40 | 954.61 | 955.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 957.40 | 954.61 | 955.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 956.20 | 954.93 | 955.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 960.00 | 954.93 | 955.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 976.85 | 959.31 | 957.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 11:15:00 | 981.05 | 966.55 | 961.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 968.60 | 969.80 | 965.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 968.60 | 969.80 | 965.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 968.60 | 969.80 | 965.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 968.60 | 969.80 | 965.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 970.65 | 969.97 | 966.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:30:00 | 970.80 | 969.97 | 966.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 961.30 | 968.23 | 965.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 961.30 | 968.23 | 965.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 955.10 | 965.61 | 964.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 955.10 | 965.61 | 964.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 962.85 | 964.97 | 964.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 966.45 | 964.97 | 964.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 958.55 | 963.69 | 964.07 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 13:15:00 | 966.85 | 964.15 | 964.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 974.20 | 966.16 | 965.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 972.55 | 976.44 | 972.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 12:15:00 | 972.55 | 976.44 | 972.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 972.55 | 976.44 | 972.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 972.55 | 976.44 | 972.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 975.75 | 976.30 | 973.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 15:00:00 | 977.50 | 976.54 | 973.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 982.25 | 976.60 | 973.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:15:00 | 978.95 | 980.11 | 977.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 970.65 | 978.40 | 978.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 970.65 | 978.40 | 978.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 964.75 | 975.67 | 977.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 969.00 | 963.62 | 967.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 969.00 | 963.62 | 967.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 969.00 | 963.62 | 967.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 971.25 | 963.62 | 967.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 969.20 | 964.74 | 967.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:45:00 | 971.60 | 964.74 | 967.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 967.40 | 965.41 | 967.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 967.40 | 965.41 | 967.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 969.85 | 966.30 | 967.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 969.50 | 966.30 | 967.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 975.35 | 968.11 | 968.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 975.35 | 968.11 | 968.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 15:15:00 | 975.10 | 969.51 | 968.99 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 962.20 | 968.04 | 968.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 945.55 | 959.39 | 963.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 958.10 | 955.57 | 960.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 958.10 | 955.57 | 960.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 958.10 | 955.57 | 960.26 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 974.25 | 963.27 | 962.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 987.55 | 975.69 | 970.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1009.30 | 1009.95 | 1000.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1009.30 | 1009.95 | 1000.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1012.70 | 1017.95 | 1011.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 1013.00 | 1017.95 | 1011.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1010.50 | 1016.46 | 1011.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1007.95 | 1016.46 | 1011.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1013.00 | 1015.77 | 1011.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:15:00 | 1013.50 | 1015.77 | 1011.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:00:00 | 1015.00 | 1015.60 | 1012.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 999.85 | 1012.12 | 1012.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 999.85 | 1012.12 | 1012.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 997.40 | 1009.18 | 1010.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1011.20 | 1007.81 | 1009.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1011.20 | 1007.81 | 1009.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1011.20 | 1007.81 | 1009.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1011.20 | 1007.81 | 1009.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1011.20 | 1008.49 | 1009.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 1013.10 | 1008.49 | 1009.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1011.25 | 1009.04 | 1009.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 1008.60 | 1008.86 | 1009.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:45:00 | 1009.10 | 1004.78 | 1006.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 1016.50 | 1008.01 | 1007.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1016.50 | 1008.01 | 1007.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1019.85 | 1014.29 | 1010.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 15:15:00 | 1060.00 | 1060.49 | 1053.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:15:00 | 1056.70 | 1060.49 | 1053.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1062.15 | 1060.82 | 1054.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 10:15:00 | 1067.75 | 1060.82 | 1054.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 1072.50 | 1063.78 | 1057.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 1067.80 | 1062.78 | 1061.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1057.30 | 1060.12 | 1060.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 1057.30 | 1060.12 | 1060.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 1054.90 | 1059.06 | 1059.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 1056.00 | 1055.96 | 1057.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 1056.00 | 1055.96 | 1057.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1056.00 | 1055.96 | 1057.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1056.00 | 1055.96 | 1057.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1055.40 | 1055.85 | 1057.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1051.80 | 1055.85 | 1057.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:00:00 | 1051.60 | 1052.59 | 1055.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 999.21 | 1029.29 | 1041.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 999.02 | 1029.29 | 1041.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 946.62 | 999.10 | 1021.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 139 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 994.65 | 967.62 | 964.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1010.90 | 987.36 | 976.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 998.50 | 1001.40 | 990.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:00:00 | 998.50 | 1001.40 | 990.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 989.20 | 1001.46 | 995.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 993.80 | 1001.46 | 995.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 995.40 | 1000.25 | 995.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 999.80 | 1000.25 | 995.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1025.90 | 1032.89 | 1033.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1025.90 | 1032.89 | 1033.71 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 1046.90 | 1035.11 | 1034.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 1056.50 | 1039.39 | 1036.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1043.20 | 1050.42 | 1044.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 1043.20 | 1050.42 | 1044.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 1042.80 | 1048.90 | 1044.67 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1039.70 | 1042.55 | 1042.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1025.10 | 1033.04 | 1037.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1033.40 | 1033.11 | 1036.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 1036.10 | 1033.11 | 1036.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1003.80 | 1027.25 | 1033.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 988.40 | 1027.25 | 1033.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 938.98 | 956.26 | 963.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 958.30 | 955.12 | 961.37 | SL hit (close>ema200) qty=0.50 sl=955.12 alert=retest2 |

### Cycle 143 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 987.50 | 966.36 | 964.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1000.90 | 976.56 | 969.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 991.90 | 991.99 | 981.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 991.90 | 991.99 | 981.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 989.80 | 995.02 | 989.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 13:00:00 | 989.80 | 995.02 | 989.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 991.90 | 994.40 | 989.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1010.50 | 992.55 | 989.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 1013.70 | 1018.34 | 1018.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-05-20 15:15:00)

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

### Cycle 145 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1011.40 | 1009.07 | 1008.81 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 1005.90 | 1008.50 | 1008.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1000.50 | 1006.90 | 1007.93 | Break + close below crossover candle low |

### Cycle 147 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1034.00 | 1010.96 | 1009.51 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 1003.00 | 1013.05 | 1014.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 1000.40 | 1010.52 | 1013.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1018.10 | 1008.29 | 1010.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 1010.00 | 1008.09 | 1010.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1003.40 | 977.85 | 976.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-06 10:15:00)

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

### Cycle 150 — SELL (started 2025-06-12 13:15:00)

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

### Cycle 151 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1008.00 | 997.36 | 996.81 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-18 10:15:00)

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

### Cycle 153 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1022.00 | 1000.03 | 997.17 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 998.65 | 999.23 | 999.25 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1002.80 | 999.94 | 999.57 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 996.50 | 999.08 | 999.27 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-06-24 09:15:00)

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

### Cycle 158 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 1016.55 | 1022.76 | 1022.98 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-07-01 12:15:00)

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

### Cycle 160 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1036.20 | 1041.41 | 1041.97 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-07-07 15:15:00)

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

### Cycle 162 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1038.50 | 1042.26 | 1042.46 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1043.20 | 1041.40 | 1041.38 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-11 12:15:00)

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

### Cycle 165 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 1037.50 | 1033.21 | 1032.87 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1028.10 | 1032.46 | 1032.60 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 1044.80 | 1034.57 | 1033.42 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1031.60 | 1032.56 | 1032.62 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-07-21 15:15:00)

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

### Cycle 170 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1029.90 | 1033.03 | 1033.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 1028.40 | 1032.00 | 1032.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1033.10 | 1031.62 | 1032.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1033.10 | 1031.62 | 1032.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-07-23 15:15:00)

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

### Cycle 172 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1022.30 | 1032.37 | 1033.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 1021.20 | 1026.78 | 1029.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 12:15:00 | 1025.70 | 1025.40 | 1028.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 1025.70 | 1025.40 | 1028.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1028.90 | 1026.10 | 1028.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:45:00 | 1029.40 | 1026.10 | 1028.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1029.10 | 1026.70 | 1028.50 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 1045.80 | 1031.05 | 1030.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1052.40 | 1043.69 | 1039.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1028.80 | 1042.67 | 1040.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1028.80 | 1042.67 | 1040.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1036.90 | 1041.51 | 1040.33 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2025-08-01 12:15:00)

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

### Cycle 175 — BUY (started 2025-08-04 11:15:00)

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

### Cycle 176 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 1044.60 | 1053.17 | 1053.37 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1057.60 | 1053.54 | 1053.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1059.60 | 1054.75 | 1053.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1053.80 | 1054.56 | 1053.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1053.80 | 1054.56 | 1053.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1053.50 | 1054.35 | 1053.76 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 1047.40 | 1052.74 | 1053.21 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1053.80 | 1053.15 | 1053.11 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1046.90 | 1052.44 | 1052.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 1041.30 | 1048.20 | 1050.55 | Break + close below crossover candle low |

### Cycle 181 — BUY (started 2025-08-18 09:15:00)

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

### Cycle 182 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 1057.00 | 1073.52 | 1074.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 1054.50 | 1069.72 | 1072.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1059.10 | 1058.71 | 1064.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 1059.10 | 1058.71 | 1064.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1056.70 | 1058.46 | 1062.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1049.70 | 1058.54 | 1062.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1047.80 | 1038.83 | 1038.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-09-02 11:15:00)

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

### Cycle 184 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 1093.60 | 1098.46 | 1098.59 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 1099.50 | 1098.47 | 1098.45 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 1094.90 | 1097.76 | 1098.13 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-09-15 11:15:00)

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

### Cycle 188 — SELL (started 2025-09-26 13:15:00)

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

### Cycle 189 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 1147.20 | 1139.17 | 1138.07 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 1135.50 | 1138.20 | 1138.40 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-01 13:15:00)

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

### Cycle 192 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 1152.40 | 1156.14 | 1156.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1142.80 | 1151.84 | 1154.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1159.40 | 1152.42 | 1154.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1159.20 | 1152.42 | 1154.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1167.90 | 1155.51 | 1155.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1170.30 | 1158.47 | 1156.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1161.00 | 1165.57 | 1161.60 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2025-10-13 12:15:00)

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

### Cycle 195 — BUY (started 2025-10-15 14:15:00)

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

### Cycle 196 — SELL (started 2025-10-20 09:15:00)

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

### Cycle 197 — BUY (started 2025-10-27 14:15:00)

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

### Cycle 198 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 1192.80 | 1200.91 | 1201.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1182.60 | 1196.36 | 1199.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1180.70 | 1169.34 | 1175.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1180.70 | 1169.34 | 1175.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1170.00 | 1169.47 | 1175.36 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1183.30 | 1176.68 | 1176.48 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1170.60 | 1175.63 | 1176.14 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 1199.80 | 1180.39 | 1178.15 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-11-14 09:15:00)

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

### Cycle 203 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1171.30 | 1168.69 | 1168.39 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-11-21 09:15:00)

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

### Cycle 205 — BUY (started 2025-11-26 12:15:00)

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

### Cycle 206 — SELL (started 2025-12-03 10:15:00)

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

### Cycle 207 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1160.70 | 1154.98 | 1154.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1168.60 | 1157.71 | 1155.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1144.00 | 1154.96 | 1154.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1144.00 | 1154.96 | 1154.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2025-12-08 10:15:00)

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

### Cycle 209 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 1128.70 | 1114.97 | 1113.21 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2025-12-16 09:15:00)

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

### Cycle 211 — BUY (started 2025-12-22 11:15:00)

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

### Cycle 212 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1082.90 | 1090.79 | 1091.87 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2025-12-26 12:15:00)

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

### Cycle 214 — SELL (started 2026-01-08 10:15:00)

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

### Cycle 215 — BUY (started 2026-01-12 14:15:00)

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

### Cycle 216 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1181.90 | 1184.11 | 1184.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1174.80 | 1182.25 | 1183.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1171.10 | 1170.59 | 1176.10 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1185.10 | 1175.92 | 1175.82 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1166.20 | 1175.98 | 1177.01 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-01-27 09:15:00)

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

### Cycle 220 — SELL (started 2026-02-01 11:15:00)

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

### Cycle 221 — BUY (started 2026-02-03 10:15:00)

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

### Cycle 222 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1226.60 | 1243.51 | 1244.51 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1247.00 | 1242.28 | 1241.74 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1237.80 | 1241.38 | 1241.38 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1245.60 | 1241.52 | 1241.35 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 1240.10 | 1241.24 | 1241.24 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-02-17 14:15:00)

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

### Cycle 228 — SELL (started 2026-02-19 11:15:00)

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

### Cycle 229 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1247.50 | 1241.83 | 1241.67 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-02-20 14:15:00)

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

### Cycle 231 — BUY (started 2026-02-24 10:15:00)

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

### Cycle 232 — SELL (started 2026-03-02 10:15:00)

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

### Cycle 233 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1242.10 | 1239.95 | 1239.92 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-03-06 14:15:00)

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

### Cycle 235 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1167.50 | 1155.03 | 1153.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1174.00 | 1161.26 | 1157.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1155.80 | 1169.05 | 1164.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1151.30 | 1169.05 | 1164.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1149.70 | 1165.18 | 1162.91 | EMA400 retest candle locked (from upside) |

### Cycle 236 — SELL (started 2026-03-19 11:15:00)

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

### Cycle 237 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 1180.30 | 1161.33 | 1159.06 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-03-23 09:15:00)

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

### Cycle 239 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1159.70 | 1141.34 | 1139.24 | EMA200 above EMA400 |

### Cycle 240 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1128.80 | 1140.75 | 1142.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1125.20 | 1135.94 | 1139.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 1133.10 | 1132.39 | 1137.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:00:00 | 1133.10 | 1132.39 | 1137.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1154.40 | 1131.97 | 1134.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1154.40 | 1131.97 | 1134.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 241 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1147.90 | 1138.35 | 1137.31 | EMA200 above EMA400 |

### Cycle 242 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1114.00 | 1135.55 | 1136.94 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-04-07 13:15:00)

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

### Cycle 244 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1256.10 | 1262.55 | 1263.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1251.60 | 1258.85 | 1261.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 1257.20 | 1257.02 | 1259.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 1257.20 | 1257.02 | 1259.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1259.30 | 1257.48 | 1259.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 1260.30 | 1257.48 | 1259.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1253.30 | 1256.64 | 1259.17 | EMA400 retest candle locked (from downside) |

### Cycle 245 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1289.90 | 1263.31 | 1261.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1301.30 | 1283.05 | 1274.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 1283.10 | 1285.82 | 1277.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:00:00 | 1283.10 | 1285.82 | 1277.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1277.60 | 1284.17 | 1277.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1278.70 | 1284.17 | 1277.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1280.30 | 1283.40 | 1278.16 | EMA400 retest candle locked (from upside) |

### Cycle 246 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1253.20 | 1274.30 | 1276.44 | EMA200 below EMA400 |

### Cycle 247 — BUY (started 2026-05-06 14:15:00)

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
| SELL | retest2 | 2023-09-18 11:30:00 | 809.70 | 2023-09-28 13:15:00 | 769.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 12:45:00 | 808.85 | 2023-09-28 13:15:00 | 768.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 14:45:00 | 809.60 | 2023-09-28 13:15:00 | 769.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 11:30:00 | 809.70 | 2023-09-29 09:15:00 | 774.90 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2023-09-18 12:45:00 | 808.85 | 2023-09-29 09:15:00 | 774.90 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2023-09-18 14:45:00 | 809.60 | 2023-09-29 09:15:00 | 774.90 | STOP_HIT | 0.50 | 4.29% |
| BUY | retest2 | 2023-10-16 11:30:00 | 787.00 | 2023-10-19 09:15:00 | 775.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-10-23 12:00:00 | 761.85 | 2023-11-01 09:15:00 | 723.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 13:00:00 | 761.65 | 2023-11-01 09:15:00 | 723.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 13:30:00 | 760.80 | 2023-11-01 09:15:00 | 723.33 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2023-10-23 12:00:00 | 761.85 | 2023-11-02 09:15:00 | 728.75 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2023-10-23 13:00:00 | 761.65 | 2023-11-02 09:15:00 | 728.75 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2023-10-23 13:30:00 | 760.80 | 2023-11-02 09:15:00 | 728.75 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2023-10-23 14:15:00 | 761.40 | 2023-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | 2.67% |
| SELL | retest2 | 2023-10-31 10:30:00 | 727.25 | 2023-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2023-11-01 09:15:00 | 727.50 | 2023-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2023-11-01 12:00:00 | 729.45 | 2023-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-11-02 10:30:00 | 729.50 | 2023-11-03 09:15:00 | 741.05 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-11-08 13:00:00 | 755.10 | 2023-11-20 13:15:00 | 763.95 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2023-11-09 10:45:00 | 753.30 | 2023-11-20 13:15:00 | 763.95 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2023-11-09 13:00:00 | 753.60 | 2023-11-20 13:15:00 | 763.95 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2023-11-09 14:00:00 | 754.40 | 2023-11-20 13:15:00 | 763.95 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2023-11-10 10:15:00 | 752.00 | 2023-11-20 13:15:00 | 763.95 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest1 | 2023-11-28 09:15:00 | 783.75 | 2023-12-04 11:15:00 | 821.84 | PARTIAL | 0.50 | 4.86% |
| BUY | retest1 | 2023-11-28 15:00:00 | 782.70 | 2023-12-04 12:15:00 | 822.94 | PARTIAL | 0.50 | 5.14% |
| BUY | retest1 | 2023-11-29 09:15:00 | 784.65 | 2023-12-05 10:15:00 | 823.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-28 09:15:00 | 783.75 | 2023-12-05 11:15:00 | 816.85 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest1 | 2023-11-28 15:00:00 | 782.70 | 2023-12-05 11:15:00 | 816.85 | STOP_HIT | 0.50 | 4.36% |
| BUY | retest1 | 2023-11-29 09:15:00 | 784.65 | 2023-12-05 11:15:00 | 816.85 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest2 | 2023-12-08 09:15:00 | 839.55 | 2023-12-19 09:15:00 | 848.30 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2023-12-22 13:15:00 | 845.20 | 2023-12-22 13:15:00 | 850.70 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-12-27 09:15:00 | 859.10 | 2024-01-02 10:15:00 | 863.55 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2024-01-19 12:15:00 | 812.50 | 2024-01-19 15:15:00 | 818.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-01-19 13:15:00 | 813.30 | 2024-01-19 15:15:00 | 818.25 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-01-20 09:45:00 | 812.95 | 2024-01-25 11:15:00 | 811.40 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-01-20 10:45:00 | 813.35 | 2024-01-25 11:15:00 | 811.40 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-01-23 10:30:00 | 805.25 | 2024-01-25 11:15:00 | 811.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-01-23 11:30:00 | 803.85 | 2024-01-25 11:15:00 | 811.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-01-25 10:45:00 | 802.95 | 2024-01-25 11:15:00 | 811.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-01-30 09:15:00 | 819.90 | 2024-02-01 12:15:00 | 808.70 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-01-31 09:30:00 | 820.90 | 2024-02-01 12:15:00 | 808.70 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-01-31 15:00:00 | 819.25 | 2024-02-01 12:15:00 | 808.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-02-12 15:15:00 | 808.25 | 2024-02-15 10:15:00 | 813.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-02-13 11:15:00 | 807.30 | 2024-02-15 10:15:00 | 813.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-02-13 12:00:00 | 808.10 | 2024-02-15 10:15:00 | 813.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-02-14 09:15:00 | 800.20 | 2024-02-15 10:15:00 | 813.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-02-20 12:45:00 | 819.25 | 2024-02-23 11:15:00 | 823.35 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-02-20 15:00:00 | 821.45 | 2024-02-23 11:15:00 | 823.35 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-02-28 10:15:00 | 812.55 | 2024-03-01 09:15:00 | 830.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-03-27 13:00:00 | 827.75 | 2024-04-05 09:15:00 | 851.10 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2024-03-27 14:30:00 | 825.35 | 2024-04-05 09:15:00 | 851.10 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2024-03-28 09:45:00 | 826.30 | 2024-04-05 09:15:00 | 851.10 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2024-04-10 09:15:00 | 878.10 | 2024-04-12 09:15:00 | 869.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-04-12 12:00:00 | 877.60 | 2024-04-12 12:15:00 | 870.15 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-04-18 14:45:00 | 845.25 | 2024-04-19 14:15:00 | 866.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-04-19 09:15:00 | 842.10 | 2024-04-19 14:15:00 | 866.20 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-04-29 09:30:00 | 891.45 | 2024-04-30 14:15:00 | 882.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-04-30 10:30:00 | 892.00 | 2024-04-30 14:15:00 | 882.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-04-30 12:15:00 | 891.80 | 2024-04-30 14:15:00 | 882.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-04-30 12:45:00 | 892.05 | 2024-04-30 14:15:00 | 882.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-05-06 15:15:00 | 880.00 | 2024-05-09 14:15:00 | 836.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 15:15:00 | 880.00 | 2024-05-10 09:15:00 | 852.60 | STOP_HIT | 0.50 | 3.11% |
| BUY | retest2 | 2024-05-16 14:15:00 | 876.80 | 2024-05-27 10:15:00 | 905.75 | STOP_HIT | 1.00 | 3.30% |
| SELL | retest2 | 2024-05-28 14:45:00 | 907.95 | 2024-06-03 10:15:00 | 912.35 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-05-29 13:45:00 | 907.35 | 2024-06-03 10:15:00 | 912.35 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-13 12:30:00 | 917.85 | 2024-06-25 15:15:00 | 929.80 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-06-13 14:45:00 | 917.05 | 2024-06-25 15:15:00 | 929.80 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2024-06-14 12:00:00 | 916.65 | 2024-06-25 15:15:00 | 929.80 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2024-06-20 09:30:00 | 917.80 | 2024-06-25 15:15:00 | 929.80 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2024-07-01 09:15:00 | 940.90 | 2024-07-08 11:15:00 | 939.10 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-09 09:15:00 | 934.95 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest1 | 2024-07-09 11:45:00 | 935.60 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest1 | 2024-07-09 13:45:00 | 935.55 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-07-11 10:15:00 | 921.50 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-07-11 11:45:00 | 920.35 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-07-11 14:30:00 | 922.15 | 2024-07-12 10:15:00 | 933.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-07-24 11:30:00 | 881.30 | 2024-07-26 09:15:00 | 895.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-07-24 14:00:00 | 882.55 | 2024-07-26 09:15:00 | 895.45 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-07-24 15:00:00 | 881.25 | 2024-07-26 09:15:00 | 895.45 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-30 10:15:00 | 903.50 | 2024-08-02 12:15:00 | 908.85 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-07-30 12:15:00 | 903.90 | 2024-08-02 12:15:00 | 908.85 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-07-30 14:00:00 | 903.15 | 2024-08-02 12:15:00 | 908.85 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-07-30 15:15:00 | 903.00 | 2024-08-02 12:15:00 | 908.85 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-09-05 11:15:00 | 935.35 | 2024-09-09 14:15:00 | 934.05 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-09-06 12:30:00 | 933.75 | 2024-09-09 14:15:00 | 934.05 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-09-11 12:00:00 | 938.80 | 2024-09-11 13:15:00 | 928.15 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest1 | 2024-09-16 09:15:00 | 960.70 | 2024-09-18 12:15:00 | 956.45 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-09-26 09:45:00 | 984.00 | 2024-10-07 11:15:00 | 1020.00 | STOP_HIT | 1.00 | 3.66% |
| SELL | retest2 | 2024-10-09 14:15:00 | 998.90 | 2024-10-10 15:15:00 | 1012.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-10-10 11:30:00 | 998.10 | 2024-10-10 15:15:00 | 1012.20 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-10-10 12:00:00 | 999.15 | 2024-10-10 15:15:00 | 1012.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-10-28 09:15:00 | 940.35 | 2024-10-28 09:15:00 | 955.85 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-11-26 14:30:00 | 964.65 | 2024-11-28 14:15:00 | 953.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-11-27 09:45:00 | 965.65 | 2024-11-28 14:15:00 | 953.55 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-11-27 12:45:00 | 963.15 | 2024-11-28 14:15:00 | 953.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-12-04 14:00:00 | 991.85 | 2024-12-05 09:15:00 | 980.95 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-04 14:30:00 | 996.10 | 2024-12-05 09:15:00 | 980.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-05 12:00:00 | 992.60 | 2024-12-11 15:15:00 | 998.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2024-12-13 09:15:00 | 982.70 | 2024-12-19 09:15:00 | 933.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 982.70 | 2024-12-20 10:15:00 | 934.10 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-01-08 15:15:00 | 900.00 | 2025-01-14 10:15:00 | 905.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-01-09 10:15:00 | 897.70 | 2025-01-14 10:15:00 | 905.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-09 12:30:00 | 900.10 | 2025-01-14 10:15:00 | 905.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-01-09 13:00:00 | 900.25 | 2025-01-14 11:15:00 | 903.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-01-13 09:15:00 | 891.90 | 2025-01-14 11:15:00 | 903.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-13 10:45:00 | 894.25 | 2025-01-14 11:15:00 | 903.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-14 10:15:00 | 895.75 | 2025-01-14 11:15:00 | 903.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-01-16 14:00:00 | 907.95 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-01-16 15:15:00 | 909.70 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-01-17 12:00:00 | 908.35 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-01-20 10:00:00 | 909.75 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-01-20 11:15:00 | 922.00 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-20 13:00:00 | 921.05 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-01-20 15:15:00 | 922.00 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-22 15:00:00 | 920.75 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-01-23 10:15:00 | 929.60 | 2025-01-27 10:15:00 | 911.40 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-02-01 10:30:00 | 951.00 | 2025-02-01 12:15:00 | 919.00 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-02-01 11:00:00 | 951.50 | 2025-02-01 12:15:00 | 919.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-02-07 09:15:00 | 960.45 | 2025-02-11 13:15:00 | 953.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-02-10 10:30:00 | 954.70 | 2025-02-11 13:15:00 | 953.45 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-02-19 15:00:00 | 977.50 | 2025-02-24 10:15:00 | 970.65 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-02-20 09:15:00 | 982.25 | 2025-02-24 10:15:00 | 970.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-02-21 10:15:00 | 978.95 | 2025-02-24 10:15:00 | 970.65 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-03-11 10:15:00 | 1013.50 | 2025-03-12 10:15:00 | 999.85 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-11 12:00:00 | 1015.00 | 2025-03-12 10:15:00 | 999.85 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-03-13 10:45:00 | 1008.60 | 2025-03-18 10:15:00 | 1016.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-03-17 09:45:00 | 1009.10 | 2025-03-18 10:15:00 | 1016.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-03-25 10:15:00 | 1067.75 | 2025-04-01 14:15:00 | 1057.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-03-25 13:45:00 | 1072.50 | 2025-04-01 14:15:00 | 1057.30 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-01 09:45:00 | 1067.80 | 2025-04-01 14:15:00 | 1057.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1051.80 | 2025-04-04 11:15:00 | 999.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 12:00:00 | 1051.60 | 2025-04-04 11:15:00 | 999.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1051.80 | 2025-04-07 09:15:00 | 946.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 12:00:00 | 1051.60 | 2025-04-07 09:15:00 | 946.44 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 11:15:00 | 999.80 | 2025-04-25 11:15:00 | 1025.90 | STOP_HIT | 1.00 | 2.61% |
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
