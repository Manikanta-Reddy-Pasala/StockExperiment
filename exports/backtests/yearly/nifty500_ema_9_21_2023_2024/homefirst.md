# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1200.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 152 |
| ALERT2 | 150 |
| ALERT2_SKIP | 85 |
| ALERT3 | 413 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 194 |
| PARTIAL | 14 |
| TARGET_HIT | 19 |
| STOP_HIT | 176 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 209 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 61 / 148
- **Target hits / Stop hits / Partials:** 19 / 176 / 14
- **Avg / median % per leg:** 0.28% / -0.77%
- **Sum % (uncompounded):** 59.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 82 | 20 | 24.4% | 14 | 68 | 0 | 0.75% | 61.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.29% | -2.3% |
| BUY @ 3rd Alert (retest2) | 81 | 20 | 24.7% | 14 | 67 | 0 | 0.78% | 63.4% |
| SELL (all) | 127 | 41 | 32.3% | 5 | 108 | 14 | -0.01% | -1.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.32% | -1.3% |
| SELL @ 3rd Alert (retest2) | 126 | 41 | 32.5% | 5 | 107 | 14 | -0.00% | -0.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.80% | -3.6% |
| retest2 (combined) | 207 | 61 | 29.5% | 19 | 174 | 14 | 0.30% | 63.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 10:15:00 | 699.80 | 715.46 | 716.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 14:15:00 | 693.05 | 705.11 | 710.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 700.95 | 697.38 | 702.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 10:00:00 | 700.95 | 697.38 | 702.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 698.85 | 697.67 | 701.81 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 708.50 | 702.74 | 702.58 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 12:15:00 | 697.75 | 702.05 | 702.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 695.00 | 699.15 | 700.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 10:15:00 | 699.85 | 699.29 | 700.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 10:30:00 | 699.00 | 699.29 | 700.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 696.05 | 698.64 | 700.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:30:00 | 696.55 | 698.64 | 700.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 698.75 | 698.47 | 699.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:00:00 | 698.75 | 698.47 | 699.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 700.60 | 698.90 | 699.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 700.60 | 698.90 | 699.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 699.05 | 698.93 | 699.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 701.05 | 698.93 | 699.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 700.25 | 699.19 | 699.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:30:00 | 693.55 | 697.27 | 698.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 14:30:00 | 695.35 | 696.45 | 697.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 12:15:00 | 695.10 | 696.39 | 697.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 11:45:00 | 694.40 | 694.25 | 695.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 696.00 | 693.38 | 694.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-05-29 09:15:00 | 706.25 | 695.96 | 695.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 706.25 | 695.96 | 695.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 13:15:00 | 707.35 | 701.95 | 698.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 699.75 | 701.51 | 698.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 699.75 | 701.51 | 698.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 699.75 | 701.51 | 698.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 15:00:00 | 699.75 | 701.51 | 698.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 698.00 | 700.81 | 698.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:15:00 | 698.00 | 700.81 | 698.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 698.90 | 700.43 | 698.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:00:00 | 704.55 | 701.25 | 699.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-05 09:15:00 | 775.00 | 749.50 | 736.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 13:15:00 | 810.20 | 823.24 | 824.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 09:15:00 | 792.95 | 814.14 | 819.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 776.35 | 759.48 | 766.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 776.35 | 759.48 | 766.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 776.35 | 759.48 | 766.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:45:00 | 787.05 | 759.48 | 766.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 756.55 | 758.89 | 765.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:15:00 | 754.00 | 758.89 | 765.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 13:00:00 | 755.00 | 758.43 | 764.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 13:30:00 | 755.25 | 757.73 | 763.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 15:00:00 | 755.05 | 757.19 | 762.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 767.45 | 758.88 | 762.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 767.45 | 758.88 | 762.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 769.75 | 761.05 | 762.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 768.05 | 761.05 | 762.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 760.60 | 761.57 | 762.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 13:30:00 | 761.15 | 761.57 | 762.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 767.95 | 762.84 | 763.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 15:00:00 | 767.95 | 762.84 | 763.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-27 15:15:00 | 770.00 | 764.27 | 763.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 15:15:00 | 770.00 | 764.27 | 763.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 783.85 | 768.19 | 765.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 12:15:00 | 786.65 | 789.09 | 781.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 13:00:00 | 786.65 | 789.09 | 781.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 786.00 | 788.35 | 782.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 15:00:00 | 786.00 | 788.35 | 782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 15:15:00 | 788.95 | 788.47 | 782.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 09:15:00 | 809.45 | 788.47 | 782.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 11:15:00 | 774.00 | 787.22 | 783.89 | SL hit (close<static) qty=1.00 sl=782.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 11:15:00 | 794.80 | 795.05 | 795.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 12:15:00 | 791.30 | 794.30 | 794.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 14:15:00 | 784.00 | 780.96 | 786.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-07 15:00:00 | 784.00 | 780.96 | 786.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 774.00 | 779.57 | 784.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 777.35 | 779.57 | 784.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 780.85 | 779.82 | 784.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 11:45:00 | 775.50 | 779.25 | 783.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 15:15:00 | 794.00 | 783.97 | 784.49 | SL hit (close>static) qty=1.00 sl=789.95 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 795.40 | 786.26 | 785.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 10:15:00 | 801.70 | 793.50 | 790.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 12:15:00 | 791.40 | 793.80 | 790.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 12:15:00 | 791.40 | 793.80 | 790.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 791.40 | 793.80 | 790.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:45:00 | 790.00 | 793.80 | 790.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 797.25 | 794.49 | 791.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 13:45:00 | 788.65 | 794.49 | 791.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 798.90 | 795.37 | 792.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:30:00 | 789.95 | 795.37 | 792.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 792.20 | 795.48 | 792.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:00:00 | 792.20 | 795.48 | 792.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 798.00 | 795.98 | 793.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:15:00 | 798.05 | 795.98 | 793.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 12:30:00 | 798.95 | 797.12 | 794.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 14:45:00 | 798.75 | 797.02 | 794.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:30:00 | 802.80 | 796.67 | 794.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 798.85 | 804.00 | 801.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:45:00 | 798.30 | 804.00 | 801.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 797.45 | 802.69 | 801.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:45:00 | 799.00 | 802.69 | 801.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 796.00 | 801.35 | 800.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 803.20 | 801.35 | 800.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 15:15:00 | 797.80 | 800.83 | 800.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 797.80 | 800.83 | 800.98 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 811.80 | 803.02 | 801.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 822.10 | 811.03 | 807.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 15:15:00 | 810.00 | 814.46 | 811.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 15:15:00 | 810.00 | 814.46 | 811.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 810.00 | 814.46 | 811.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 09:15:00 | 825.05 | 814.46 | 811.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 13:15:00 | 848.45 | 858.24 | 859.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 848.45 | 858.24 | 859.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 14:15:00 | 840.95 | 854.79 | 857.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 09:15:00 | 851.45 | 851.10 | 855.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 09:45:00 | 853.90 | 851.10 | 855.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 848.00 | 850.48 | 854.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:30:00 | 847.20 | 850.48 | 854.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 839.85 | 841.59 | 847.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:30:00 | 839.10 | 841.59 | 847.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 834.55 | 837.44 | 842.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:30:00 | 831.45 | 836.69 | 841.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 11:30:00 | 833.00 | 835.93 | 840.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:45:00 | 830.80 | 834.31 | 839.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 14:45:00 | 833.15 | 834.28 | 838.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 852.00 | 837.94 | 839.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-07 09:15:00 | 852.00 | 837.94 | 839.49 | SL hit (close>static) qty=1.00 sl=851.20 alert=retest2 |

### Cycle 12 — BUY (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 11:15:00 | 849.80 | 842.15 | 841.24 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 839.10 | 846.50 | 846.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 13:15:00 | 836.50 | 841.39 | 843.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 14:15:00 | 843.90 | 841.90 | 843.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 14:15:00 | 843.90 | 841.90 | 843.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 843.90 | 841.90 | 843.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 843.90 | 841.90 | 843.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 844.85 | 842.49 | 843.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 843.80 | 842.49 | 843.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 853.00 | 844.59 | 844.64 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 10:15:00 | 852.60 | 846.19 | 845.36 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 15:15:00 | 841.25 | 846.77 | 847.40 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 910.15 | 859.45 | 853.10 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 14:15:00 | 857.35 | 867.86 | 867.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 12:15:00 | 853.70 | 862.18 | 864.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 825.00 | 820.56 | 827.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 825.00 | 820.56 | 827.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 825.00 | 820.56 | 827.53 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 839.45 | 831.46 | 830.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 10:15:00 | 844.90 | 834.15 | 831.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 10:15:00 | 850.60 | 852.38 | 844.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 10:45:00 | 848.70 | 852.38 | 844.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 844.95 | 850.29 | 845.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 844.95 | 850.29 | 845.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 844.00 | 849.03 | 845.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:30:00 | 843.40 | 849.03 | 845.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 846.00 | 848.42 | 845.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 855.10 | 848.42 | 845.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 11:15:00 | 857.80 | 860.73 | 860.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 857.80 | 860.73 | 860.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 12:15:00 | 854.60 | 858.50 | 859.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 09:15:00 | 862.00 | 857.27 | 858.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 862.00 | 857.27 | 858.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 862.00 | 857.27 | 858.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:00:00 | 862.00 | 857.27 | 858.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 853.05 | 856.43 | 857.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 11:30:00 | 847.80 | 854.64 | 856.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 15:15:00 | 847.10 | 856.12 | 857.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 09:45:00 | 842.00 | 851.11 | 854.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 12:15:00 | 850.25 | 850.77 | 853.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 852.70 | 846.20 | 849.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:00:00 | 852.70 | 846.20 | 849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 853.35 | 847.63 | 850.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:15:00 | 858.85 | 847.63 | 850.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 850.50 | 850.20 | 850.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:30:00 | 847.00 | 850.20 | 850.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 851.25 | 850.41 | 850.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 851.25 | 850.41 | 850.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 853.25 | 850.98 | 851.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 10:00:00 | 853.25 | 850.98 | 851.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 844.00 | 849.58 | 850.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 863.40 | 851.87 | 850.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 09:15:00 | 863.40 | 851.87 | 850.96 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 841.00 | 850.02 | 850.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 838.70 | 845.00 | 847.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 14:15:00 | 838.00 | 836.23 | 838.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 14:15:00 | 838.00 | 836.23 | 838.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 838.00 | 836.23 | 838.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 838.00 | 836.23 | 838.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 839.70 | 836.93 | 838.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 837.00 | 836.93 | 838.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 835.00 | 836.54 | 838.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 10:45:00 | 829.40 | 835.26 | 837.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:30:00 | 824.45 | 826.69 | 831.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 11:15:00 | 842.55 | 831.11 | 830.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 11:15:00 | 842.55 | 831.11 | 830.40 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 821.40 | 830.33 | 830.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 15:15:00 | 820.00 | 828.26 | 829.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 09:15:00 | 825.65 | 818.23 | 822.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 825.65 | 818.23 | 822.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 825.65 | 818.23 | 822.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 825.65 | 818.23 | 822.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 823.85 | 819.36 | 822.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:15:00 | 825.40 | 819.36 | 822.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 825.00 | 820.49 | 822.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:45:00 | 825.90 | 820.49 | 822.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 13:15:00 | 832.30 | 824.23 | 824.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 15:15:00 | 835.00 | 827.31 | 825.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 826.80 | 827.21 | 825.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 826.80 | 827.21 | 825.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 826.80 | 827.21 | 825.67 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 817.50 | 825.16 | 825.32 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 834.90 | 826.52 | 825.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 11:15:00 | 840.05 | 831.23 | 828.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 14:15:00 | 842.60 | 846.37 | 840.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 14:15:00 | 842.60 | 846.37 | 840.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 842.60 | 846.37 | 840.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:00:00 | 842.60 | 846.37 | 840.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 848.80 | 846.86 | 841.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 835.70 | 846.86 | 841.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 844.10 | 846.31 | 841.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 10:15:00 | 845.85 | 846.31 | 841.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 11:30:00 | 845.90 | 846.32 | 842.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 13:00:00 | 845.75 | 846.21 | 842.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 14:00:00 | 845.40 | 846.05 | 842.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 14:15:00 | 845.00 | 845.84 | 843.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 14:45:00 | 844.70 | 845.84 | 843.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 845.25 | 845.72 | 843.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 850.10 | 845.72 | 843.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 13:30:00 | 847.00 | 851.02 | 850.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-17 09:15:00 | 930.44 | 902.35 | 883.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 931.50 | 943.97 | 945.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 921.15 | 937.55 | 941.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 915.75 | 891.77 | 903.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 13:15:00 | 915.75 | 891.77 | 903.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 915.75 | 891.77 | 903.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:00:00 | 915.75 | 891.77 | 903.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 906.50 | 894.72 | 903.99 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 929.50 | 908.94 | 908.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 12:15:00 | 946.00 | 920.52 | 914.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 13:15:00 | 937.95 | 943.18 | 932.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 14:00:00 | 937.95 | 943.18 | 932.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 936.80 | 941.91 | 933.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 15:00:00 | 936.80 | 941.91 | 933.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 934.90 | 940.50 | 933.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:30:00 | 926.25 | 936.59 | 932.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 911.00 | 931.47 | 930.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:00:00 | 911.00 | 931.47 | 930.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 11:15:00 | 910.70 | 927.32 | 928.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 13:15:00 | 905.45 | 919.87 | 924.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 15:15:00 | 914.00 | 913.40 | 917.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 09:15:00 | 916.35 | 913.40 | 917.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 912.25 | 913.17 | 916.79 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 10:15:00 | 923.00 | 916.63 | 916.55 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 15:15:00 | 915.00 | 916.69 | 916.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 13:15:00 | 909.95 | 914.45 | 915.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 915.60 | 911.88 | 913.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 915.60 | 911.88 | 913.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 915.60 | 911.88 | 913.86 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 920.40 | 914.88 | 914.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 923.50 | 917.54 | 915.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 13:15:00 | 920.55 | 921.80 | 918.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 13:15:00 | 920.55 | 921.80 | 918.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 920.55 | 921.80 | 918.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:00:00 | 920.55 | 921.80 | 918.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 920.90 | 921.62 | 918.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:45:00 | 920.00 | 921.62 | 918.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 954.95 | 933.36 | 926.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 10:45:00 | 960.95 | 948.25 | 939.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 11:15:00 | 960.05 | 948.25 | 939.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 13:15:00 | 939.95 | 953.25 | 953.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 939.95 | 953.25 | 953.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 14:15:00 | 936.00 | 949.80 | 951.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 947.20 | 945.31 | 949.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 947.20 | 945.31 | 949.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 947.20 | 945.31 | 949.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 957.00 | 945.31 | 949.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 945.00 | 945.25 | 948.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 945.00 | 945.25 | 948.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 939.75 | 942.33 | 945.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 13:00:00 | 936.55 | 940.06 | 943.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 09:15:00 | 936.50 | 939.78 | 942.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 09:15:00 | 889.72 | 916.20 | 927.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 09:15:00 | 889.67 | 916.20 | 927.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-24 10:15:00 | 890.90 | 889.07 | 903.85 | SL hit (close>ema200) qty=0.50 sl=889.07 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 893.35 | 879.47 | 879.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 13:15:00 | 900.80 | 883.74 | 881.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 931.15 | 949.49 | 937.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 931.15 | 949.49 | 937.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 931.15 | 949.49 | 937.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:00:00 | 931.15 | 949.49 | 937.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 933.30 | 946.25 | 936.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:45:00 | 925.40 | 946.25 | 936.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 946.70 | 946.44 | 939.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 14:00:00 | 946.70 | 946.44 | 939.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 1029.40 | 1026.39 | 1007.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:45:00 | 1012.00 | 1026.39 | 1007.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 1024.75 | 1025.28 | 1013.69 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 994.00 | 1012.65 | 1012.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 985.00 | 1007.12 | 1010.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 999.95 | 994.92 | 1001.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 999.95 | 994.92 | 1001.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 999.95 | 994.92 | 1001.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:45:00 | 1010.10 | 994.92 | 1001.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 1005.35 | 997.00 | 1001.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:45:00 | 1005.15 | 997.00 | 1001.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 1010.00 | 999.60 | 1002.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:30:00 | 1011.85 | 999.60 | 1002.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 1011.00 | 1004.47 | 1004.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 1048.95 | 1013.37 | 1008.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 10:15:00 | 1021.00 | 1028.39 | 1021.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 10:15:00 | 1021.00 | 1028.39 | 1021.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 1021.00 | 1028.39 | 1021.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:00:00 | 1021.00 | 1028.39 | 1021.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 1023.25 | 1027.36 | 1021.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 12:15:00 | 1024.70 | 1027.36 | 1021.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 13:45:00 | 1025.00 | 1025.69 | 1021.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 14:15:00 | 1024.80 | 1025.69 | 1021.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 14:45:00 | 1024.40 | 1024.96 | 1021.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 1026.50 | 1025.27 | 1022.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 1032.40 | 1025.27 | 1022.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-19 09:15:00 | 1021.25 | 1024.47 | 1022.08 | SL hit (close<static) qty=1.00 sl=1021.90 alert=retest2 |

### Cycle 37 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 11:15:00 | 1006.80 | 1020.37 | 1020.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 12:15:00 | 1004.25 | 1017.15 | 1019.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 1014.05 | 1009.98 | 1014.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 1014.05 | 1009.98 | 1014.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1014.05 | 1009.98 | 1014.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 11:15:00 | 998.85 | 1009.03 | 1013.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 13:15:00 | 948.91 | 958.25 | 964.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-01 09:15:00 | 939.40 | 938.96 | 947.85 | SL hit (close>ema200) qty=0.50 sl=938.96 alert=retest2 |

### Cycle 38 — BUY (started 2024-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 15:15:00 | 950.00 | 943.01 | 942.29 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-01-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 15:15:00 | 944.00 | 947.16 | 947.58 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 956.90 | 949.11 | 948.42 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 942.90 | 948.32 | 948.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 938.50 | 945.02 | 947.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 941.50 | 940.97 | 943.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 941.50 | 940.97 | 943.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 941.50 | 940.97 | 943.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 11:30:00 | 936.05 | 939.82 | 942.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 14:00:00 | 934.05 | 940.27 | 941.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 943.00 | 941.91 | 941.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 943.00 | 941.91 | 941.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 12:15:00 | 949.45 | 943.42 | 942.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 964.95 | 968.09 | 960.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 10:00:00 | 964.95 | 968.09 | 960.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 974.95 | 969.46 | 961.83 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 14:15:00 | 953.65 | 962.02 | 962.70 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 10:15:00 | 975.10 | 965.12 | 963.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 11:15:00 | 993.90 | 973.81 | 969.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 1000.00 | 1002.62 | 992.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 1000.00 | 1002.62 | 992.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1000.00 | 1002.62 | 992.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:45:00 | 999.85 | 1002.62 | 992.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 992.40 | 998.66 | 993.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:30:00 | 991.80 | 998.66 | 993.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 986.45 | 996.22 | 992.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:45:00 | 985.10 | 996.22 | 992.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 985.85 | 994.14 | 991.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 985.85 | 994.14 | 991.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 995.00 | 993.71 | 992.20 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 984.00 | 991.23 | 991.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 958.95 | 978.37 | 983.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 991.45 | 973.38 | 978.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 991.45 | 973.38 | 978.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 991.45 | 973.38 | 978.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:30:00 | 993.50 | 973.38 | 978.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 992.00 | 977.10 | 979.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:45:00 | 990.50 | 977.10 | 979.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 994.05 | 982.87 | 981.68 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 977.00 | 982.61 | 982.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 973.50 | 980.04 | 981.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 989.40 | 981.28 | 981.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 989.40 | 981.28 | 981.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 989.40 | 981.28 | 981.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:15:00 | 989.05 | 981.28 | 981.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 984.10 | 981.84 | 982.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 11:15:00 | 983.60 | 981.84 | 982.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 11:15:00 | 984.85 | 982.44 | 982.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 984.85 | 982.44 | 982.26 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 13:15:00 | 979.50 | 982.21 | 982.21 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 986.90 | 983.15 | 982.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 15:15:00 | 989.70 | 984.46 | 983.28 | Break + close above crossover candle high |

### Cycle 51 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 965.60 | 983.75 | 984.32 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 985.85 | 975.80 | 975.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 11:15:00 | 1004.15 | 983.36 | 979.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 994.30 | 996.76 | 992.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 994.30 | 996.76 | 992.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 994.30 | 996.76 | 992.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:30:00 | 994.00 | 996.76 | 992.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 994.95 | 995.90 | 992.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:30:00 | 994.00 | 995.90 | 992.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 993.50 | 995.15 | 992.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 13:45:00 | 993.25 | 995.15 | 992.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 990.60 | 994.24 | 992.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:45:00 | 982.95 | 994.24 | 992.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 984.90 | 992.37 | 991.79 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 972.85 | 988.47 | 990.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 10:15:00 | 963.00 | 983.37 | 987.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 955.25 | 954.44 | 964.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 955.25 | 954.44 | 964.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 939.95 | 935.63 | 942.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:30:00 | 940.65 | 935.63 | 942.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 939.45 | 936.40 | 942.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:00:00 | 939.45 | 936.40 | 942.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 940.30 | 937.18 | 942.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:45:00 | 940.30 | 937.18 | 942.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 940.10 | 938.26 | 942.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 14:00:00 | 940.10 | 938.26 | 942.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 900.45 | 930.88 | 937.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 11:00:00 | 893.10 | 923.32 | 933.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 11:45:00 | 894.35 | 917.59 | 930.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 14:30:00 | 893.95 | 905.28 | 920.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 09:15:00 | 926.40 | 922.15 | 921.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 926.40 | 922.15 | 921.93 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 918.40 | 921.36 | 921.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 912.55 | 919.60 | 920.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 931.30 | 920.70 | 921.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 931.30 | 920.70 | 921.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 931.30 | 920.70 | 921.08 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 10:15:00 | 934.35 | 923.43 | 922.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 11:15:00 | 936.90 | 926.12 | 923.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 932.55 | 948.39 | 941.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 932.55 | 948.39 | 941.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 932.55 | 948.39 | 941.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 931.40 | 948.39 | 941.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 929.55 | 944.62 | 940.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:15:00 | 927.15 | 944.62 | 940.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 927.30 | 937.77 | 938.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 11:15:00 | 919.95 | 928.39 | 932.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 875.55 | 874.53 | 887.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 12:15:00 | 892.40 | 879.55 | 886.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 892.40 | 879.55 | 886.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:00:00 | 892.40 | 879.55 | 886.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 893.45 | 882.33 | 887.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:30:00 | 894.80 | 882.33 | 887.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-03-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 15:15:00 | 916.00 | 893.86 | 892.01 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 871.50 | 891.75 | 893.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 869.00 | 887.20 | 891.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 850.80 | 846.94 | 860.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 10:00:00 | 850.80 | 846.94 | 860.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 874.00 | 854.05 | 860.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:00:00 | 874.00 | 854.05 | 860.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 885.00 | 860.24 | 863.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:30:00 | 898.15 | 860.24 | 863.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 866.45 | 863.02 | 863.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 855.00 | 863.02 | 863.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:15:00 | 812.25 | 823.35 | 829.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-15 12:15:00 | 823.80 | 823.44 | 828.83 | SL hit (close>ema200) qty=0.50 sl=823.44 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 841.50 | 831.90 | 830.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 12:15:00 | 847.90 | 835.10 | 832.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 15:15:00 | 847.70 | 849.17 | 843.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 15:15:00 | 847.70 | 849.17 | 843.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 847.70 | 849.17 | 843.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:15:00 | 857.90 | 849.17 | 843.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-02 14:15:00 | 943.69 | 929.80 | 918.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 934.95 | 949.60 | 951.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 925.65 | 941.37 | 946.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 13:15:00 | 893.60 | 892.89 | 904.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 14:15:00 | 894.70 | 892.89 | 904.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 892.00 | 892.72 | 903.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 892.00 | 892.72 | 903.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 902.85 | 894.74 | 903.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 10:15:00 | 884.75 | 893.41 | 901.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:00:00 | 882.10 | 889.54 | 895.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:30:00 | 884.80 | 887.97 | 894.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 12:15:00 | 884.65 | 887.97 | 894.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 885.00 | 881.45 | 884.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 10:45:00 | 879.25 | 880.56 | 883.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 14:15:00 | 875.00 | 871.57 | 871.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 14:15:00 | 879.70 | 873.19 | 872.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 879.70 | 873.19 | 872.52 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 866.20 | 871.45 | 872.09 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 875.00 | 872.67 | 872.50 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 869.80 | 872.09 | 872.25 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 09:15:00 | 880.00 | 873.32 | 872.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 13:15:00 | 888.05 | 879.36 | 876.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 15:15:00 | 880.00 | 880.39 | 877.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 11:15:00 | 905.70 | 881.94 | 878.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 885.00 | 897.45 | 889.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 885.00 | 897.45 | 889.60 | SL hit (close<ema400) qty=1.00 sl=889.60 alert=retest1 |

### Cycle 67 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 885.40 | 895.86 | 896.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 870.05 | 889.65 | 893.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 886.00 | 883.96 | 888.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 886.00 | 883.96 | 888.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 886.00 | 883.96 | 888.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:30:00 | 886.50 | 883.96 | 888.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 886.55 | 884.48 | 888.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 886.55 | 884.48 | 888.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 891.20 | 885.83 | 888.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 883.00 | 890.20 | 890.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 838.85 | 859.36 | 872.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-05-13 12:15:00 | 794.70 | 812.34 | 834.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 826.80 | 822.07 | 821.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 840.00 | 829.57 | 825.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 11:15:00 | 829.95 | 831.11 | 827.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 829.95 | 831.11 | 827.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 827.35 | 830.36 | 827.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:15:00 | 822.30 | 830.36 | 827.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 806.30 | 825.55 | 825.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 806.30 | 825.55 | 825.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 829.85 | 826.41 | 825.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 833.00 | 826.41 | 825.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 12:15:00 | 819.35 | 832.77 | 833.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 819.35 | 832.77 | 833.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 818.00 | 827.69 | 830.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 827.35 | 827.16 | 829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 827.35 | 827.16 | 829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 827.35 | 827.16 | 829.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:45:00 | 813.45 | 818.64 | 821.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 815.05 | 812.77 | 816.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 814.30 | 814.38 | 816.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 814.95 | 818.03 | 818.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 814.95 | 817.42 | 817.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 819.85 | 817.42 | 817.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 814.70 | 816.87 | 817.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 811.00 | 815.91 | 817.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 812.00 | 811.53 | 813.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 813.15 | 812.18 | 813.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 12:45:00 | 813.20 | 812.60 | 813.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 813.05 | 812.69 | 813.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 816.25 | 812.69 | 813.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 814.05 | 812.96 | 813.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 814.05 | 812.96 | 813.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 814.25 | 813.22 | 813.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 802.20 | 813.22 | 813.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 816.50 | 812.10 | 811.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 816.50 | 812.10 | 811.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 843.15 | 819.42 | 815.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 818.80 | 827.93 | 823.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 818.80 | 827.93 | 823.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 818.80 | 827.93 | 823.00 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 806.10 | 818.90 | 819.47 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 849.30 | 818.00 | 816.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 928.15 | 873.45 | 860.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1064.75 | 1067.61 | 1045.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 1064.65 | 1067.61 | 1045.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1044.55 | 1066.07 | 1056.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1044.55 | 1066.07 | 1056.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1048.00 | 1062.45 | 1055.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:30:00 | 1038.75 | 1062.45 | 1055.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 1037.30 | 1050.99 | 1051.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 1030.60 | 1042.12 | 1045.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 15:15:00 | 1024.00 | 1018.76 | 1026.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 15:15:00 | 1024.00 | 1018.76 | 1026.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1024.00 | 1018.76 | 1026.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1040.75 | 1018.76 | 1026.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1026.50 | 1020.31 | 1026.95 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 1041.40 | 1030.50 | 1030.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 1049.15 | 1036.10 | 1033.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 1035.20 | 1037.30 | 1034.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 1035.20 | 1037.30 | 1034.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1035.20 | 1037.30 | 1034.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 1035.20 | 1037.30 | 1034.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1034.10 | 1036.66 | 1034.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 1033.05 | 1036.66 | 1034.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1029.90 | 1035.31 | 1034.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 1029.95 | 1035.31 | 1034.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1020.70 | 1032.39 | 1032.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 09:15:00 | 1015.25 | 1025.72 | 1029.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 1030.45 | 1024.98 | 1027.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 1030.45 | 1024.98 | 1027.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1030.45 | 1024.98 | 1027.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 1035.45 | 1024.98 | 1027.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1041.65 | 1028.31 | 1028.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1041.65 | 1028.31 | 1028.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1028.10 | 1028.27 | 1028.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1048.10 | 1028.27 | 1028.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1082.65 | 1039.14 | 1033.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 1136.90 | 1058.70 | 1043.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 1093.55 | 1102.84 | 1077.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:45:00 | 1095.05 | 1102.84 | 1077.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1098.45 | 1112.09 | 1101.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 1098.45 | 1112.09 | 1101.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1097.05 | 1109.08 | 1101.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 1098.40 | 1109.08 | 1101.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 1094.10 | 1106.09 | 1100.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 1094.10 | 1106.09 | 1100.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1083.00 | 1101.47 | 1098.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:00:00 | 1083.00 | 1101.47 | 1098.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 1078.20 | 1096.81 | 1096.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 14:15:00 | 1066.25 | 1090.70 | 1094.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 1076.10 | 1063.77 | 1073.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1076.10 | 1063.77 | 1073.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1076.10 | 1063.77 | 1073.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 1076.10 | 1063.77 | 1073.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1060.55 | 1063.13 | 1072.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:45:00 | 1049.95 | 1060.02 | 1070.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:45:00 | 1051.95 | 1058.60 | 1068.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1094.00 | 1053.68 | 1056.69 | SL hit (close>static) qty=1.00 sl=1086.95 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 1075.25 | 1059.46 | 1057.45 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 1045.30 | 1055.75 | 1056.15 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 1059.95 | 1056.59 | 1056.50 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 1054.95 | 1056.26 | 1056.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 1045.30 | 1054.07 | 1055.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 13:15:00 | 1059.90 | 1052.42 | 1053.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 13:15:00 | 1059.90 | 1052.42 | 1053.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 1059.90 | 1052.42 | 1053.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 1059.90 | 1052.42 | 1053.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 1058.00 | 1053.54 | 1054.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:15:00 | 1063.00 | 1053.54 | 1054.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 15:15:00 | 1063.00 | 1055.43 | 1055.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 1091.00 | 1065.95 | 1060.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 1093.85 | 1094.29 | 1080.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:45:00 | 1094.75 | 1094.29 | 1080.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1076.30 | 1090.69 | 1080.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1076.30 | 1090.69 | 1080.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1071.40 | 1086.84 | 1079.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 1071.40 | 1086.84 | 1079.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1082.50 | 1085.97 | 1079.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1092.10 | 1085.97 | 1079.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:45:00 | 1090.45 | 1094.56 | 1088.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 1083.35 | 1091.24 | 1088.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 15:15:00 | 1074.05 | 1084.59 | 1085.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 1074.05 | 1084.59 | 1085.91 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 1106.45 | 1088.96 | 1087.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 1116.00 | 1095.30 | 1090.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 1098.05 | 1106.58 | 1100.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 1098.05 | 1106.58 | 1100.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1098.05 | 1106.58 | 1100.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 1098.05 | 1106.58 | 1100.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1100.00 | 1105.27 | 1100.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1084.65 | 1105.27 | 1100.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1074.15 | 1099.04 | 1097.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1074.15 | 1099.04 | 1097.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 1065.85 | 1092.40 | 1094.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 12:15:00 | 1054.30 | 1072.48 | 1082.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 1099.70 | 1052.20 | 1058.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 1099.70 | 1052.20 | 1058.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1099.70 | 1052.20 | 1058.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 1101.15 | 1052.20 | 1058.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1093.00 | 1060.36 | 1062.06 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 1085.35 | 1065.36 | 1064.17 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 1060.00 | 1064.09 | 1064.10 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1073.50 | 1065.97 | 1064.95 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 1055.25 | 1065.37 | 1066.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 1051.15 | 1062.52 | 1064.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 15:15:00 | 1060.35 | 1059.62 | 1062.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 1046.60 | 1059.62 | 1062.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1057.85 | 1059.26 | 1062.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1034.85 | 1045.26 | 1052.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:45:00 | 1035.15 | 1041.68 | 1049.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:45:00 | 1035.00 | 1041.38 | 1048.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:15:00 | 1033.90 | 1041.38 | 1048.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1027.10 | 1027.40 | 1034.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1007.65 | 1027.40 | 1034.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 10:30:00 | 1023.00 | 1023.86 | 1031.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 13:15:00 | 983.11 | 1009.36 | 1022.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 13:15:00 | 983.39 | 1009.36 | 1022.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 13:15:00 | 983.25 | 1009.36 | 1022.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 13:15:00 | 982.21 | 1009.36 | 1022.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 1036.50 | 1008.91 | 1018.39 | SL hit (close>ema200) qty=0.50 sl=1008.91 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 14:15:00 | 1040.00 | 1023.92 | 1022.90 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 11:15:00 | 1016.50 | 1021.61 | 1022.29 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 1028.65 | 1022.65 | 1022.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1046.25 | 1027.75 | 1024.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1021.20 | 1036.14 | 1032.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1021.20 | 1036.14 | 1032.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1021.20 | 1036.14 | 1032.13 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 1002.80 | 1026.56 | 1028.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 990.60 | 1006.91 | 1012.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1016.20 | 1007.65 | 1011.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1016.20 | 1007.65 | 1011.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1016.20 | 1007.65 | 1011.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 1016.60 | 1007.65 | 1011.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1011.45 | 1008.41 | 1011.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 12:30:00 | 1007.50 | 1010.15 | 1012.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:30:00 | 1010.45 | 1010.26 | 1011.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:15:00 | 1010.00 | 1010.26 | 1011.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 1034.40 | 1013.96 | 1013.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1034.40 | 1013.96 | 1013.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 1059.60 | 1034.55 | 1026.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 1052.10 | 1053.81 | 1039.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 1052.10 | 1053.81 | 1039.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1045.65 | 1053.76 | 1046.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 1045.65 | 1053.76 | 1046.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1048.60 | 1052.73 | 1046.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 1046.65 | 1052.73 | 1046.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1050.00 | 1052.18 | 1046.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 1052.05 | 1052.18 | 1046.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:45:00 | 1051.25 | 1051.59 | 1047.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-30 09:15:00 | 1157.26 | 1118.16 | 1106.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 1119.75 | 1130.27 | 1130.92 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 1149.80 | 1132.53 | 1131.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 1169.50 | 1144.03 | 1137.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 1157.90 | 1169.73 | 1158.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 12:15:00 | 1157.90 | 1169.73 | 1158.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1157.90 | 1169.73 | 1158.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 1157.90 | 1169.73 | 1158.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 1152.20 | 1166.22 | 1157.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 1151.55 | 1166.22 | 1157.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1143.85 | 1161.75 | 1156.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 1142.20 | 1161.75 | 1156.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1108.35 | 1148.31 | 1151.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 1099.15 | 1132.67 | 1143.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 09:15:00 | 1086.70 | 1080.61 | 1099.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 1111.95 | 1086.88 | 1100.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1111.95 | 1086.88 | 1100.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 1111.95 | 1086.88 | 1100.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 1102.95 | 1090.10 | 1100.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:15:00 | 1100.45 | 1090.10 | 1100.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 1100.00 | 1092.27 | 1100.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 1100.70 | 1095.35 | 1101.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 1138.45 | 1109.59 | 1106.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 1138.45 | 1109.59 | 1106.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 1158.60 | 1119.39 | 1111.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 14:15:00 | 1124.85 | 1128.15 | 1118.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 15:00:00 | 1124.85 | 1128.15 | 1118.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1179.70 | 1194.32 | 1182.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1166.50 | 1194.32 | 1182.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1151.65 | 1185.79 | 1179.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1151.65 | 1185.79 | 1179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1139.15 | 1176.46 | 1176.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 1139.15 | 1176.46 | 1176.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 1147.95 | 1170.76 | 1173.70 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1233.05 | 1180.52 | 1174.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1281.95 | 1230.24 | 1210.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1305.00 | 1322.12 | 1299.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 1305.00 | 1322.12 | 1299.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1302.25 | 1318.14 | 1299.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 1299.80 | 1318.14 | 1299.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1294.75 | 1313.47 | 1299.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:30:00 | 1296.30 | 1313.47 | 1299.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1298.75 | 1310.52 | 1299.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1305.00 | 1310.52 | 1299.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 1286.85 | 1302.52 | 1298.18 | SL hit (close<static) qty=1.00 sl=1290.60 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1278.10 | 1294.19 | 1294.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 1262.30 | 1284.16 | 1289.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 1186.95 | 1184.09 | 1206.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 1186.95 | 1184.09 | 1206.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1175.00 | 1178.24 | 1190.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1211.95 | 1178.24 | 1190.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1255.05 | 1193.60 | 1196.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 1255.05 | 1193.60 | 1196.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1247.00 | 1204.28 | 1201.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-07 09:15:00 | 1291.55 | 1243.40 | 1224.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 12:15:00 | 1252.00 | 1252.79 | 1234.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 13:00:00 | 1252.00 | 1252.79 | 1234.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1235.00 | 1249.46 | 1235.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 1235.00 | 1249.46 | 1235.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 1250.00 | 1249.57 | 1237.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:15:00 | 1277.90 | 1249.57 | 1237.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1269.55 | 1253.56 | 1240.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 1339.00 | 1270.74 | 1255.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 12:15:00 | 1244.95 | 1275.05 | 1278.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1244.95 | 1275.05 | 1278.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 1241.05 | 1263.14 | 1271.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 1216.35 | 1210.62 | 1227.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 14:15:00 | 1216.35 | 1210.62 | 1227.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1216.35 | 1210.62 | 1227.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 1216.35 | 1210.62 | 1227.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1201.75 | 1201.61 | 1211.77 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 11:15:00 | 1228.75 | 1212.92 | 1211.82 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1199.95 | 1209.74 | 1210.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1178.30 | 1203.45 | 1207.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1123.35 | 1116.34 | 1135.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 1123.35 | 1116.34 | 1135.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1116.15 | 1115.64 | 1128.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1116.15 | 1115.64 | 1128.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1109.35 | 1113.56 | 1124.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 1093.35 | 1109.52 | 1121.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1205.85 | 1130.02 | 1125.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 09:15:00 | 1205.85 | 1130.02 | 1125.26 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1110.60 | 1121.86 | 1122.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 1095.80 | 1114.18 | 1118.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 13:15:00 | 1106.85 | 1106.49 | 1111.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 13:45:00 | 1105.60 | 1106.49 | 1111.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 108 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 1162.65 | 1118.19 | 1115.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 1175.60 | 1131.19 | 1122.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 1196.15 | 1205.84 | 1180.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:15:00 | 1192.30 | 1205.84 | 1180.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1195.05 | 1203.68 | 1181.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 1185.95 | 1203.68 | 1181.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1188.70 | 1198.48 | 1184.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1184.50 | 1198.48 | 1184.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1179.00 | 1194.58 | 1183.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 1179.00 | 1194.58 | 1183.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1180.00 | 1191.66 | 1183.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 1178.00 | 1191.66 | 1183.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1185.00 | 1190.16 | 1184.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1185.00 | 1190.16 | 1184.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1156.45 | 1183.41 | 1182.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1156.45 | 1183.41 | 1182.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1148.80 | 1176.49 | 1179.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 1143.70 | 1165.69 | 1173.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1161.30 | 1157.76 | 1166.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1161.30 | 1157.76 | 1166.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1161.30 | 1157.76 | 1166.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 14:45:00 | 1147.80 | 1157.53 | 1163.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 1147.30 | 1154.75 | 1161.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1167.00 | 1162.60 | 1162.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 1167.00 | 1162.60 | 1162.43 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1148.30 | 1159.74 | 1161.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 1144.45 | 1156.68 | 1159.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 1043.00 | 1035.15 | 1052.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:15:00 | 1043.90 | 1035.15 | 1052.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1060.55 | 1040.23 | 1052.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1060.55 | 1040.23 | 1052.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1050.25 | 1042.24 | 1052.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1065.50 | 1042.24 | 1052.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1054.70 | 1044.73 | 1052.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 1052.95 | 1044.73 | 1052.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1059.90 | 1047.76 | 1053.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1059.90 | 1047.76 | 1053.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1048.40 | 1047.89 | 1053.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 1042.50 | 1047.89 | 1053.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 1041.80 | 1048.71 | 1052.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 1045.00 | 1047.76 | 1051.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 1043.90 | 1046.99 | 1050.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1051.30 | 1046.42 | 1049.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 1051.30 | 1046.42 | 1049.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1052.00 | 1047.53 | 1049.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 1065.35 | 1047.53 | 1049.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1073.05 | 1052.64 | 1051.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1073.05 | 1052.64 | 1051.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 1079.20 | 1063.09 | 1058.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 1106.95 | 1120.04 | 1103.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 1106.95 | 1120.04 | 1103.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1106.95 | 1120.04 | 1103.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 1101.15 | 1120.04 | 1103.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1108.60 | 1117.76 | 1103.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:45:00 | 1113.15 | 1114.57 | 1105.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 1091.55 | 1108.50 | 1104.95 | SL hit (close<static) qty=1.00 sl=1103.10 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 1097.80 | 1102.91 | 1103.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 1095.00 | 1101.33 | 1102.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1101.05 | 1100.05 | 1101.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 1101.05 | 1100.05 | 1101.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1101.05 | 1100.05 | 1101.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 1102.10 | 1100.05 | 1101.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1097.45 | 1099.53 | 1101.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1098.00 | 1099.53 | 1101.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1109.85 | 1101.60 | 1101.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1109.85 | 1101.60 | 1101.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1099.60 | 1101.20 | 1101.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 15:00:00 | 1091.40 | 1099.33 | 1100.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1087.40 | 1099.47 | 1100.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 09:15:00 | 982.26 | 1054.41 | 1073.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 1103.00 | 1069.75 | 1068.77 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 09:15:00 | 1055.25 | 1068.67 | 1070.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1040.45 | 1046.34 | 1051.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 1023.65 | 1021.63 | 1030.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 10:45:00 | 1024.35 | 1021.63 | 1030.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1025.45 | 1021.79 | 1028.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 1027.35 | 1021.79 | 1028.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1029.75 | 1023.38 | 1028.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 1029.75 | 1023.38 | 1028.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1031.30 | 1024.97 | 1029.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1031.30 | 1024.97 | 1029.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1033.90 | 1026.75 | 1029.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:00:00 | 1027.30 | 1026.86 | 1029.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 1036.00 | 1029.27 | 1030.04 | SL hit (close>static) qty=1.00 sl=1035.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 14:15:00 | 1040.15 | 1032.25 | 1031.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 1052.90 | 1039.39 | 1035.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 1034.80 | 1039.37 | 1036.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1034.80 | 1039.37 | 1036.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1034.80 | 1039.37 | 1036.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 1034.55 | 1039.37 | 1036.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1035.55 | 1038.60 | 1036.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 1042.70 | 1039.42 | 1036.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 11:45:00 | 1040.05 | 1041.93 | 1040.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 1028.85 | 1039.31 | 1039.18 | SL hit (close<static) qty=1.00 sl=1030.10 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1023.30 | 1036.11 | 1037.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 1020.05 | 1033.26 | 1036.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 1020.55 | 1018.14 | 1024.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 12:00:00 | 1020.55 | 1018.14 | 1024.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1023.30 | 1019.72 | 1024.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1023.30 | 1019.72 | 1024.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1031.20 | 1022.02 | 1025.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 1031.20 | 1022.02 | 1025.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1028.00 | 1023.21 | 1025.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 1027.80 | 1023.21 | 1025.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1022.05 | 1022.98 | 1025.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:30:00 | 1017.65 | 1022.07 | 1024.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 1024.90 | 1006.80 | 1005.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 1024.90 | 1006.80 | 1005.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 1086.00 | 1033.25 | 1019.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 14:15:00 | 1055.70 | 1058.26 | 1045.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 15:00:00 | 1055.70 | 1058.26 | 1045.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 1054.10 | 1054.65 | 1049.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 1060.70 | 1053.01 | 1049.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:00:00 | 1060.55 | 1065.84 | 1064.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 1038.40 | 1060.35 | 1062.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1038.40 | 1060.35 | 1062.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1029.05 | 1054.09 | 1059.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1060.05 | 1049.25 | 1054.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 1060.05 | 1049.25 | 1054.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1060.05 | 1049.25 | 1054.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1060.05 | 1049.25 | 1054.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1058.45 | 1051.09 | 1055.00 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 1060.00 | 1057.58 | 1057.33 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1052.55 | 1056.57 | 1056.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 13:15:00 | 1026.00 | 1046.31 | 1051.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 1048.00 | 1046.65 | 1051.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 1048.00 | 1046.65 | 1051.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1048.00 | 1046.65 | 1051.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1048.00 | 1046.65 | 1051.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1060.30 | 1049.38 | 1052.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1057.50 | 1049.38 | 1052.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1061.05 | 1051.71 | 1052.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1050.00 | 1053.10 | 1053.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 1058.05 | 1054.09 | 1053.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 1058.05 | 1054.09 | 1053.88 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 1035.35 | 1050.60 | 1052.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1019.45 | 1041.48 | 1047.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 14:15:00 | 1038.95 | 1035.21 | 1041.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 14:15:00 | 1038.95 | 1035.21 | 1041.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1038.95 | 1035.21 | 1041.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:45:00 | 1042.55 | 1035.21 | 1041.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 1022.05 | 1018.81 | 1027.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:15:00 | 1019.90 | 1018.81 | 1027.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1019.85 | 1019.02 | 1026.86 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 1040.05 | 1029.92 | 1029.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 1065.60 | 1038.67 | 1033.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 12:15:00 | 1041.50 | 1042.53 | 1037.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 13:00:00 | 1041.50 | 1042.53 | 1037.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 1037.85 | 1041.59 | 1037.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:30:00 | 1039.10 | 1041.59 | 1037.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1029.95 | 1039.26 | 1036.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 1029.95 | 1039.26 | 1036.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 1030.80 | 1037.57 | 1036.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 1044.55 | 1037.57 | 1036.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 1044.20 | 1042.55 | 1042.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 1025.95 | 1043.61 | 1044.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1025.95 | 1043.61 | 1044.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1017.45 | 1038.38 | 1042.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 997.30 | 996.08 | 1007.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 14:00:00 | 997.30 | 996.08 | 1007.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 989.15 | 994.46 | 1003.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 999.40 | 994.46 | 1003.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1003.05 | 996.74 | 1002.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 1003.05 | 996.74 | 1002.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 995.75 | 996.54 | 1001.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 992.90 | 996.71 | 1001.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 971.30 | 996.46 | 1000.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 943.25 | 969.61 | 981.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 962.70 | 961.19 | 970.60 | SL hit (close>ema200) qty=0.50 sl=961.19 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 984.00 | 970.91 | 969.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 1001.70 | 987.43 | 978.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 990.45 | 991.86 | 982.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 990.45 | 991.86 | 982.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 990.45 | 991.86 | 982.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 993.05 | 991.86 | 982.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 960.90 | 985.67 | 980.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 960.90 | 985.67 | 980.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 968.50 | 982.23 | 979.37 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 962.00 | 975.37 | 976.57 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 989.90 | 978.28 | 977.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 13:15:00 | 997.90 | 986.97 | 982.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 999.00 | 1000.39 | 995.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 13:30:00 | 999.00 | 1000.39 | 995.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1029.00 | 1006.73 | 999.57 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 996.70 | 1006.52 | 1007.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 990.80 | 1001.78 | 1004.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 967.95 | 963.68 | 976.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 948.70 | 963.68 | 976.88 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 961.20 | 950.45 | 958.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 961.20 | 950.45 | 958.35 | SL hit (close>ema400) qty=1.00 sl=958.35 alert=retest1 |

### Cycle 130 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 944.85 | 933.69 | 933.24 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 918.00 | 933.17 | 934.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 907.60 | 923.17 | 929.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 922.70 | 902.82 | 910.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 922.70 | 902.82 | 910.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 922.70 | 902.82 | 910.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 922.70 | 902.82 | 910.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 924.40 | 907.13 | 912.14 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 924.75 | 916.67 | 915.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 10:15:00 | 943.75 | 924.23 | 919.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 10:15:00 | 967.50 | 973.83 | 952.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:00:00 | 967.50 | 973.83 | 952.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 963.95 | 971.85 | 953.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:45:00 | 954.55 | 971.85 | 953.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1007.95 | 1005.89 | 995.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 1001.45 | 1005.89 | 995.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 1001.85 | 1004.23 | 996.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:30:00 | 997.50 | 1004.23 | 996.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 991.50 | 1001.45 | 997.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 1002.20 | 1001.45 | 997.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 997.65 | 1000.69 | 997.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 1007.10 | 1000.69 | 997.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1004.90 | 1003.95 | 1001.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:45:00 | 1004.15 | 1004.47 | 1002.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:30:00 | 1006.30 | 1004.78 | 1002.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1002.25 | 1004.28 | 1002.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 1002.25 | 1004.28 | 1002.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 998.20 | 1003.06 | 1002.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:15:00 | 995.10 | 1003.06 | 1002.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1004.75 | 1003.40 | 1002.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:15:00 | 996.00 | 1003.40 | 1002.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 996.00 | 1001.92 | 1001.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 982.80 | 1001.92 | 1001.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 997.45 | 1001.03 | 1001.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 997.45 | 1001.03 | 1001.40 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 1006.20 | 1002.06 | 1001.84 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 992.60 | 1000.17 | 1001.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 988.55 | 997.84 | 999.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 1006.25 | 994.91 | 997.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 1006.25 | 994.91 | 997.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1006.25 | 994.91 | 997.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 1008.50 | 994.91 | 997.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 998.95 | 995.72 | 997.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 999.25 | 995.72 | 997.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1000.15 | 996.61 | 997.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 1000.15 | 996.61 | 997.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 998.85 | 997.06 | 997.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 998.85 | 997.06 | 997.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1000.00 | 997.64 | 998.02 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 999.95 | 998.33 | 998.28 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 995.95 | 998.02 | 998.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 993.65 | 996.68 | 997.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1004.85 | 996.69 | 997.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1004.85 | 996.69 | 997.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1004.85 | 996.69 | 997.05 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 1009.65 | 999.28 | 998.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 1012.85 | 1002.00 | 999.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 14:15:00 | 1002.00 | 1003.68 | 1001.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-17 15:00:00 | 1002.00 | 1003.68 | 1001.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 999.10 | 1002.76 | 1000.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 986.40 | 1002.76 | 1000.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 992.00 | 1000.61 | 1000.07 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 10:15:00 | 990.00 | 998.49 | 999.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-18 12:15:00 | 977.45 | 992.17 | 996.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 991.55 | 990.42 | 994.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 14:45:00 | 991.50 | 990.42 | 994.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 995.30 | 991.39 | 994.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 1006.10 | 991.39 | 994.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 999.65 | 993.05 | 995.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:30:00 | 1002.85 | 993.05 | 995.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 998.80 | 994.20 | 995.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:30:00 | 1000.65 | 994.20 | 995.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 988.85 | 994.39 | 995.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:45:00 | 992.35 | 994.39 | 995.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 993.00 | 994.12 | 995.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:15:00 | 999.35 | 994.12 | 995.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 987.00 | 992.69 | 994.34 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 1005.00 | 996.88 | 995.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 10:15:00 | 1052.90 | 1011.05 | 1002.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1062.80 | 1075.50 | 1056.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1062.80 | 1075.50 | 1056.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1055.65 | 1068.73 | 1056.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1058.20 | 1068.73 | 1056.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1039.35 | 1062.85 | 1055.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 1034.55 | 1062.85 | 1055.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1037.25 | 1057.73 | 1053.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:15:00 | 1033.00 | 1057.73 | 1053.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1023.25 | 1046.42 | 1048.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1011.15 | 1039.37 | 1045.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 1026.90 | 1021.58 | 1030.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 1026.90 | 1021.58 | 1030.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1026.90 | 1021.58 | 1030.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 1030.55 | 1021.58 | 1030.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1043.35 | 1025.93 | 1031.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1036.45 | 1025.93 | 1031.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1038.05 | 1028.35 | 1032.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 1034.40 | 1028.35 | 1032.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 1048.25 | 1035.43 | 1034.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1048.25 | 1035.43 | 1034.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 1065.00 | 1041.34 | 1037.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 10:15:00 | 1038.20 | 1042.29 | 1038.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 1038.20 | 1042.29 | 1038.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1038.20 | 1042.29 | 1038.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 1038.20 | 1042.29 | 1038.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1015.95 | 1037.02 | 1036.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 1015.95 | 1037.02 | 1036.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 1017.55 | 1033.13 | 1035.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 1012.25 | 1028.95 | 1032.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 1002.05 | 994.22 | 1005.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 1002.05 | 994.22 | 1005.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1000.00 | 996.44 | 1002.49 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 1020.40 | 1006.65 | 1005.70 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 1000.00 | 1006.83 | 1007.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 963.80 | 998.15 | 1003.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 12:15:00 | 991.00 | 988.29 | 996.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 12:30:00 | 990.05 | 988.29 | 996.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 995.95 | 989.82 | 996.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:00:00 | 995.95 | 989.82 | 996.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 1009.80 | 993.82 | 997.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 1009.80 | 993.82 | 997.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1007.80 | 996.61 | 998.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 1002.25 | 996.61 | 998.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 995.20 | 996.33 | 998.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 987.45 | 995.81 | 997.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 984.75 | 988.40 | 993.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 09:15:00 | 1030.10 | 996.84 | 996.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 1030.10 | 996.84 | 996.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 1060.90 | 1017.93 | 1007.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 10:15:00 | 1181.50 | 1186.64 | 1160.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 11:00:00 | 1181.50 | 1186.64 | 1160.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1261.10 | 1271.50 | 1254.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:30:00 | 1260.70 | 1271.50 | 1254.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1255.00 | 1268.20 | 1254.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1255.00 | 1268.20 | 1254.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1255.20 | 1265.60 | 1254.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1250.60 | 1265.60 | 1254.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 1266.00 | 1265.68 | 1255.61 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 09:15:00 | 1242.50 | 1250.39 | 1250.70 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1265.00 | 1249.56 | 1249.29 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 1241.60 | 1253.72 | 1254.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 1228.40 | 1244.76 | 1250.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1154.00 | 1152.56 | 1166.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 1154.00 | 1152.56 | 1166.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1170.30 | 1154.92 | 1165.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 1170.30 | 1154.92 | 1165.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1174.40 | 1158.81 | 1166.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 1165.50 | 1165.00 | 1167.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 1165.90 | 1157.55 | 1159.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 1165.60 | 1157.55 | 1159.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 12:00:00 | 1164.60 | 1158.96 | 1159.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 1165.40 | 1161.39 | 1160.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1165.40 | 1161.39 | 1160.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1182.70 | 1169.32 | 1166.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 1173.20 | 1174.16 | 1170.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:30:00 | 1177.30 | 1174.16 | 1170.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1173.10 | 1173.95 | 1170.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 1174.10 | 1173.95 | 1170.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1171.70 | 1173.50 | 1170.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 1170.20 | 1173.50 | 1170.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1178.00 | 1174.40 | 1171.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:30:00 | 1176.30 | 1174.40 | 1171.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1163.00 | 1173.00 | 1171.41 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 1168.00 | 1170.32 | 1170.44 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1218.90 | 1178.92 | 1174.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1226.70 | 1188.47 | 1178.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1194.30 | 1203.57 | 1192.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1194.30 | 1203.57 | 1192.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1194.30 | 1203.57 | 1192.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1194.10 | 1203.57 | 1192.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1197.00 | 1202.26 | 1193.24 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1180.40 | 1189.15 | 1189.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1157.70 | 1175.55 | 1181.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 1163.30 | 1158.35 | 1167.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:45:00 | 1160.00 | 1158.35 | 1167.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1162.90 | 1158.93 | 1166.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1162.90 | 1158.93 | 1166.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1168.90 | 1159.02 | 1163.05 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1175.70 | 1165.40 | 1164.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 1186.00 | 1171.90 | 1168.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 1154.30 | 1180.95 | 1176.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 1154.30 | 1180.95 | 1176.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1154.30 | 1180.95 | 1176.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 1154.30 | 1180.95 | 1176.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1159.00 | 1176.56 | 1174.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 1165.40 | 1176.56 | 1174.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 13:15:00 | 1281.94 | 1238.90 | 1211.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 1277.20 | 1284.44 | 1284.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 1269.90 | 1281.53 | 1283.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 1279.40 | 1274.07 | 1278.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 1279.40 | 1274.07 | 1278.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1279.40 | 1274.07 | 1278.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 1280.00 | 1274.07 | 1278.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1284.80 | 1276.21 | 1278.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 1284.80 | 1276.21 | 1278.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1275.10 | 1275.99 | 1278.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 1270.90 | 1275.99 | 1278.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:30:00 | 1273.30 | 1270.88 | 1274.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1271.00 | 1261.67 | 1261.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 1271.00 | 1261.67 | 1261.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 1278.00 | 1267.83 | 1264.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 1269.90 | 1270.54 | 1266.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:30:00 | 1271.90 | 1270.54 | 1266.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1260.20 | 1268.47 | 1266.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:45:00 | 1260.00 | 1268.47 | 1266.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1274.90 | 1269.76 | 1267.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 1276.90 | 1271.20 | 1268.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:00:00 | 1276.00 | 1271.20 | 1268.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:45:00 | 1280.00 | 1273.36 | 1269.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 1277.70 | 1278.90 | 1273.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1269.60 | 1277.04 | 1273.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 1269.60 | 1277.04 | 1273.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1269.70 | 1275.57 | 1272.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:15:00 | 1265.90 | 1275.57 | 1272.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1285.00 | 1277.46 | 1274.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1293.80 | 1278.57 | 1274.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 1288.50 | 1280.71 | 1276.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 1291.80 | 1284.37 | 1278.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 1290.00 | 1284.32 | 1279.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1281.00 | 1285.35 | 1281.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 1268.20 | 1285.35 | 1281.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1282.70 | 1284.82 | 1281.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 1286.30 | 1284.82 | 1281.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1275.10 | 1282.87 | 1280.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1275.10 | 1282.87 | 1280.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1268.00 | 1279.90 | 1279.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1268.00 | 1279.90 | 1279.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 1264.80 | 1276.88 | 1278.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 1264.80 | 1276.88 | 1278.21 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1295.40 | 1279.18 | 1278.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 1302.50 | 1283.84 | 1281.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 1293.00 | 1297.10 | 1290.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 15:15:00 | 1293.00 | 1297.10 | 1290.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1293.00 | 1297.10 | 1290.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1342.10 | 1297.10 | 1290.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 14:15:00 | 1476.31 | 1388.29 | 1369.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1322.50 | 1371.74 | 1377.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 1316.30 | 1352.63 | 1367.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 1365.00 | 1343.87 | 1357.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 1365.00 | 1343.87 | 1357.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1365.00 | 1343.87 | 1357.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1365.00 | 1343.87 | 1357.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1353.00 | 1345.69 | 1356.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 1349.00 | 1346.56 | 1356.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 15:15:00 | 1348.00 | 1348.40 | 1354.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 10:00:00 | 1347.00 | 1348.05 | 1353.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 1361.50 | 1356.31 | 1356.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 1361.50 | 1356.31 | 1356.15 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 1348.00 | 1355.27 | 1355.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 1322.40 | 1347.14 | 1351.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 1345.70 | 1341.70 | 1347.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 1345.70 | 1341.70 | 1347.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1345.70 | 1341.70 | 1347.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 1345.70 | 1341.70 | 1347.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 1395.60 | 1352.48 | 1351.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 1431.60 | 1377.84 | 1369.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 1379.10 | 1382.29 | 1374.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:30:00 | 1380.70 | 1382.29 | 1374.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1381.00 | 1382.03 | 1375.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1381.00 | 1382.03 | 1375.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1365.00 | 1378.63 | 1374.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1378.80 | 1378.63 | 1374.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1392.70 | 1381.44 | 1376.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1398.10 | 1381.44 | 1376.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:45:00 | 1396.50 | 1383.99 | 1378.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 1400.00 | 1385.28 | 1380.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 1398.00 | 1396.84 | 1391.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1384.00 | 1394.28 | 1390.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 1404.40 | 1394.28 | 1390.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 1397.70 | 1412.45 | 1413.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1397.70 | 1412.45 | 1413.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 1391.00 | 1405.32 | 1409.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1373.00 | 1371.91 | 1383.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 1373.00 | 1371.91 | 1383.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1399.10 | 1377.64 | 1381.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1399.10 | 1377.64 | 1381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1394.10 | 1380.93 | 1382.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 1396.50 | 1380.93 | 1382.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 1399.50 | 1384.65 | 1384.27 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 1380.00 | 1386.09 | 1386.11 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 1404.20 | 1389.63 | 1387.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 15:15:00 | 1409.20 | 1393.55 | 1389.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 1401.80 | 1430.83 | 1420.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1401.80 | 1430.83 | 1420.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1401.80 | 1430.83 | 1420.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 1389.80 | 1430.83 | 1420.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1391.90 | 1423.05 | 1417.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:45:00 | 1389.60 | 1423.05 | 1417.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1394.10 | 1413.12 | 1413.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1379.50 | 1406.40 | 1410.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 1205.50 | 1203.82 | 1230.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 1205.50 | 1203.82 | 1230.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1231.80 | 1212.77 | 1230.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1231.80 | 1212.77 | 1230.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1241.00 | 1218.42 | 1231.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1241.00 | 1218.42 | 1231.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1256.10 | 1225.95 | 1233.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1256.10 | 1225.95 | 1233.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 1275.80 | 1241.37 | 1239.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 1280.70 | 1262.27 | 1251.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1234.10 | 1259.47 | 1252.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1234.10 | 1259.47 | 1252.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1234.10 | 1259.47 | 1252.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1234.10 | 1259.47 | 1252.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1229.20 | 1253.42 | 1250.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1229.20 | 1253.42 | 1250.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 1230.80 | 1245.55 | 1246.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1218.20 | 1236.74 | 1242.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1223.00 | 1222.15 | 1230.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1210.00 | 1222.15 | 1230.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1206.10 | 1218.94 | 1228.42 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1282.10 | 1232.39 | 1227.76 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 1224.90 | 1241.37 | 1242.60 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 1249.90 | 1240.59 | 1239.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 1252.30 | 1242.93 | 1240.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 1256.00 | 1256.01 | 1248.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 1256.00 | 1256.01 | 1248.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1252.00 | 1256.09 | 1250.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1252.00 | 1256.09 | 1250.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1265.00 | 1257.87 | 1252.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:00:00 | 1278.90 | 1260.81 | 1255.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 1268.00 | 1263.44 | 1261.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1270.30 | 1268.16 | 1264.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1265.00 | 1296.28 | 1299.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1265.00 | 1296.28 | 1299.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 1256.80 | 1283.06 | 1292.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 1226.50 | 1223.59 | 1240.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1247.50 | 1228.73 | 1237.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1247.50 | 1228.73 | 1237.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1246.00 | 1228.73 | 1237.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1251.10 | 1233.20 | 1238.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1251.10 | 1233.20 | 1238.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 13:15:00 | 1247.60 | 1242.10 | 1241.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1258.80 | 1246.62 | 1244.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 1252.00 | 1255.00 | 1250.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:30:00 | 1257.50 | 1255.00 | 1250.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1251.30 | 1254.26 | 1250.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:30:00 | 1249.70 | 1254.26 | 1250.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1250.70 | 1253.55 | 1250.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1250.70 | 1253.55 | 1250.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1250.10 | 1252.86 | 1250.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 1258.40 | 1252.86 | 1250.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 1254.40 | 1260.22 | 1260.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1254.40 | 1260.22 | 1260.67 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1274.50 | 1262.24 | 1261.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 1295.00 | 1279.01 | 1271.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1297.30 | 1298.78 | 1290.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1297.30 | 1298.78 | 1290.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1292.90 | 1297.95 | 1291.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 1307.90 | 1300.74 | 1296.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1280.00 | 1293.75 | 1294.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 1280.00 | 1293.75 | 1294.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1274.30 | 1289.86 | 1292.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 1300.70 | 1281.91 | 1285.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1300.70 | 1281.91 | 1285.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1300.70 | 1281.91 | 1285.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 1279.20 | 1283.09 | 1285.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 1277.80 | 1282.04 | 1284.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1268.90 | 1282.23 | 1284.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 1277.50 | 1269.57 | 1268.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 1277.50 | 1269.57 | 1268.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 12:15:00 | 1281.90 | 1272.04 | 1269.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1266.00 | 1274.26 | 1272.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1266.00 | 1274.26 | 1272.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1266.00 | 1274.26 | 1272.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 1262.70 | 1274.26 | 1272.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 1255.20 | 1270.44 | 1270.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 1248.20 | 1266.00 | 1268.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1222.00 | 1205.29 | 1225.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1222.00 | 1205.29 | 1225.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1222.00 | 1205.29 | 1225.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1222.00 | 1205.29 | 1225.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1208.40 | 1205.92 | 1223.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 1193.90 | 1203.68 | 1217.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1203.30 | 1203.83 | 1213.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:45:00 | 1205.90 | 1205.77 | 1212.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 1202.00 | 1208.67 | 1212.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1221.90 | 1210.98 | 1212.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1221.90 | 1210.98 | 1212.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1235.40 | 1215.86 | 1214.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1235.40 | 1215.86 | 1214.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1254.60 | 1226.51 | 1219.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1239.60 | 1242.42 | 1233.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 1239.60 | 1242.42 | 1233.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1239.60 | 1242.42 | 1233.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 1239.60 | 1242.42 | 1233.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1236.40 | 1241.21 | 1233.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 1236.40 | 1241.21 | 1233.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1230.00 | 1238.97 | 1233.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1229.90 | 1238.97 | 1233.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1228.20 | 1236.82 | 1232.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 1227.00 | 1236.82 | 1232.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1221.30 | 1231.44 | 1231.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 1221.30 | 1231.44 | 1231.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 1209.90 | 1227.13 | 1229.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 1207.20 | 1223.14 | 1227.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 11:15:00 | 1216.40 | 1214.04 | 1219.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 12:00:00 | 1216.40 | 1214.04 | 1219.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1216.40 | 1213.86 | 1218.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 1211.50 | 1213.86 | 1218.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1219.50 | 1215.30 | 1218.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1217.50 | 1215.30 | 1218.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1212.80 | 1214.80 | 1217.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 1210.20 | 1213.88 | 1217.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 1210.50 | 1213.34 | 1216.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1211.00 | 1207.77 | 1211.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1219.90 | 1214.81 | 1214.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1219.90 | 1214.81 | 1214.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1227.10 | 1218.19 | 1216.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 14:15:00 | 1219.10 | 1221.08 | 1218.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 1219.10 | 1221.08 | 1218.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1219.10 | 1221.08 | 1218.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 1219.50 | 1221.08 | 1218.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1216.80 | 1220.23 | 1218.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 1217.30 | 1219.04 | 1218.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1220.00 | 1219.23 | 1218.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1228.00 | 1219.88 | 1218.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 1229.40 | 1242.38 | 1242.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1229.40 | 1242.38 | 1242.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1222.70 | 1236.75 | 1239.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 1219.30 | 1219.05 | 1227.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:45:00 | 1223.30 | 1219.05 | 1227.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1227.90 | 1220.82 | 1227.51 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1253.00 | 1234.36 | 1232.09 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 1223.10 | 1230.13 | 1230.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1217.00 | 1227.51 | 1229.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1227.30 | 1225.62 | 1228.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1227.30 | 1225.62 | 1228.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1227.30 | 1225.62 | 1228.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 1227.30 | 1225.62 | 1228.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1227.70 | 1226.04 | 1228.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 1223.00 | 1225.43 | 1227.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:00:00 | 1221.80 | 1224.71 | 1227.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:30:00 | 1220.60 | 1220.98 | 1224.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 1212.20 | 1208.20 | 1207.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1212.20 | 1208.20 | 1207.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 1215.00 | 1209.56 | 1208.45 | Break + close above crossover candle high |

### Cycle 187 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1192.20 | 1207.24 | 1207.79 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1222.50 | 1208.43 | 1208.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 11:15:00 | 1233.00 | 1216.06 | 1211.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1232.90 | 1236.96 | 1226.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 1232.90 | 1236.96 | 1226.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1232.90 | 1236.96 | 1226.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1232.90 | 1236.96 | 1226.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1226.00 | 1234.77 | 1226.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 1223.00 | 1234.77 | 1226.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 1204.90 | 1228.79 | 1224.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 1204.90 | 1228.79 | 1224.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1201.50 | 1223.34 | 1222.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1205.10 | 1223.34 | 1222.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1197.40 | 1218.15 | 1220.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1094.30 | 1189.05 | 1206.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1137.90 | 1132.67 | 1153.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:00:00 | 1137.90 | 1132.67 | 1153.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1166.00 | 1141.45 | 1150.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1166.00 | 1141.45 | 1150.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1171.30 | 1147.42 | 1152.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 1171.30 | 1147.42 | 1152.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1172.40 | 1156.22 | 1156.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 1176.40 | 1160.26 | 1157.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1162.00 | 1163.09 | 1159.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 10:00:00 | 1162.00 | 1163.09 | 1159.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1156.10 | 1162.51 | 1160.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1156.10 | 1162.51 | 1160.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1162.50 | 1162.51 | 1160.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 1151.90 | 1162.51 | 1160.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1176.40 | 1165.29 | 1162.05 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1149.80 | 1158.89 | 1159.77 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 1163.40 | 1157.85 | 1157.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 1171.10 | 1160.50 | 1158.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1182.90 | 1188.69 | 1177.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 1182.90 | 1188.69 | 1177.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1210.00 | 1192.96 | 1180.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 1215.80 | 1202.47 | 1188.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 1190.90 | 1197.78 | 1197.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 1190.90 | 1197.78 | 1197.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 1187.60 | 1194.94 | 1196.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1192.60 | 1189.80 | 1193.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 13:15:00 | 1192.60 | 1189.80 | 1193.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1192.60 | 1189.80 | 1193.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 1192.60 | 1189.80 | 1193.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1186.10 | 1189.06 | 1192.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1178.30 | 1188.85 | 1192.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 15:15:00 | 1119.38 | 1151.26 | 1165.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1143.20 | 1132.25 | 1145.51 | SL hit (close>ema200) qty=0.50 sl=1132.25 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 1109.30 | 1095.21 | 1094.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 1119.10 | 1101.94 | 1098.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1180.10 | 1184.12 | 1172.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:45:00 | 1177.40 | 1184.12 | 1172.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1169.40 | 1180.43 | 1172.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 1169.40 | 1180.43 | 1172.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1169.40 | 1178.23 | 1172.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 14:15:00 | 1174.20 | 1178.23 | 1172.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 14:45:00 | 1175.20 | 1177.48 | 1172.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1166.50 | 1179.02 | 1179.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1166.50 | 1179.02 | 1179.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1163.90 | 1172.98 | 1176.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1145.10 | 1139.38 | 1150.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 1145.10 | 1139.38 | 1150.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1145.10 | 1139.38 | 1150.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1145.10 | 1139.38 | 1150.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1170.00 | 1145.50 | 1152.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1134.20 | 1145.50 | 1152.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 1141.50 | 1145.02 | 1150.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1118.90 | 1108.47 | 1107.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1118.90 | 1108.47 | 1107.61 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 1104.90 | 1107.82 | 1107.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 1095.80 | 1105.42 | 1106.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 15:15:00 | 1062.50 | 1062.33 | 1073.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:15:00 | 1055.00 | 1062.33 | 1073.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1049.70 | 1059.80 | 1071.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 1043.20 | 1055.36 | 1068.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1061.00 | 1045.97 | 1044.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1061.00 | 1045.97 | 1044.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 1063.80 | 1049.54 | 1046.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 1084.60 | 1084.63 | 1074.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 13:45:00 | 1082.60 | 1084.63 | 1074.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1074.20 | 1081.73 | 1078.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1069.20 | 1081.73 | 1078.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1075.40 | 1080.46 | 1078.51 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1069.80 | 1076.67 | 1077.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 1065.20 | 1072.46 | 1074.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1044.10 | 1043.43 | 1052.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1044.10 | 1043.43 | 1052.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1042.30 | 1043.21 | 1051.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1047.40 | 1043.21 | 1051.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1039.00 | 1042.36 | 1050.17 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 1107.00 | 1056.83 | 1053.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1202.60 | 1116.21 | 1088.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 1177.00 | 1197.47 | 1153.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:45:00 | 1183.50 | 1197.47 | 1153.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1176.00 | 1193.18 | 1155.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 1157.60 | 1193.18 | 1155.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1187.90 | 1206.53 | 1195.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 1187.90 | 1206.53 | 1195.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1183.50 | 1201.92 | 1193.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 1183.50 | 1201.92 | 1193.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1177.10 | 1196.96 | 1192.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 1176.90 | 1196.96 | 1192.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 1166.20 | 1185.23 | 1187.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1156.80 | 1173.52 | 1181.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1178.00 | 1142.04 | 1150.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1178.00 | 1142.04 | 1150.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1178.00 | 1142.04 | 1150.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 1160.00 | 1147.12 | 1152.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 1163.50 | 1156.31 | 1155.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 1163.50 | 1156.31 | 1155.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 1183.10 | 1163.54 | 1159.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1164.00 | 1176.21 | 1169.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1164.00 | 1176.21 | 1169.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1164.00 | 1176.21 | 1169.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1160.10 | 1176.21 | 1169.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1160.10 | 1172.99 | 1168.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1160.10 | 1172.99 | 1168.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1154.20 | 1169.23 | 1167.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:45:00 | 1154.70 | 1169.23 | 1167.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1158.80 | 1164.77 | 1165.31 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1172.20 | 1165.07 | 1164.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1175.00 | 1168.37 | 1166.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 12:15:00 | 1174.50 | 1179.63 | 1173.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 12:15:00 | 1174.50 | 1179.63 | 1173.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1174.50 | 1179.63 | 1173.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 1174.50 | 1179.63 | 1173.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1171.90 | 1178.09 | 1173.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 1187.30 | 1174.95 | 1172.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 1177.90 | 1178.65 | 1177.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 1177.60 | 1178.65 | 1177.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1178.30 | 1182.76 | 1180.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1179.30 | 1182.07 | 1180.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1186.90 | 1183.65 | 1181.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1184.60 | 1183.56 | 1182.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1188.90 | 1183.99 | 1182.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 1171.00 | 1181.73 | 1181.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 1171.00 | 1181.73 | 1181.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 13:15:00 | 1169.10 | 1177.93 | 1180.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 1173.20 | 1171.85 | 1175.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:30:00 | 1173.50 | 1171.85 | 1175.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1184.50 | 1174.38 | 1176.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 1188.10 | 1174.38 | 1176.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1187.60 | 1177.03 | 1177.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1185.60 | 1177.03 | 1177.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 1183.00 | 1178.68 | 1178.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 1185.40 | 1179.58 | 1178.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 14:15:00 | 1178.10 | 1182.22 | 1180.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1178.10 | 1182.22 | 1180.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1178.10 | 1182.22 | 1180.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1178.10 | 1182.22 | 1180.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1179.00 | 1181.58 | 1180.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 1173.60 | 1181.58 | 1180.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1176.00 | 1180.46 | 1179.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1176.00 | 1180.46 | 1179.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1172.20 | 1178.81 | 1179.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1170.00 | 1177.05 | 1178.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 1162.80 | 1159.43 | 1166.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:45:00 | 1163.00 | 1159.43 | 1166.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1172.30 | 1162.00 | 1167.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1172.30 | 1162.00 | 1167.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1158.00 | 1161.20 | 1166.50 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 1171.50 | 1168.63 | 1168.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 1188.40 | 1173.70 | 1170.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 1183.60 | 1219.91 | 1206.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1183.60 | 1219.91 | 1206.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1183.60 | 1219.91 | 1206.69 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1182.30 | 1197.28 | 1198.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 14:15:00 | 1175.60 | 1192.94 | 1196.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 11:15:00 | 1059.80 | 1054.01 | 1071.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:45:00 | 1063.10 | 1054.01 | 1071.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1073.30 | 1047.20 | 1053.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1073.30 | 1047.20 | 1053.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1102.80 | 1058.32 | 1057.86 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1054.80 | 1066.54 | 1066.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 1050.00 | 1063.23 | 1065.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1006.30 | 993.64 | 1002.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1006.30 | 993.64 | 1002.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1006.30 | 993.64 | 1002.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1006.30 | 993.64 | 1002.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 998.00 | 994.51 | 1002.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 994.50 | 994.51 | 1002.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 996.90 | 993.93 | 1000.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 944.77 | 959.81 | 971.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 947.05 | 959.81 | 971.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 11:15:00 | 897.21 | 925.40 | 946.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 212 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 958.70 | 933.78 | 932.01 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 927.00 | 935.41 | 936.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 913.50 | 931.03 | 934.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 936.70 | 918.29 | 923.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 936.70 | 918.29 | 923.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 936.70 | 918.29 | 923.73 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 944.85 | 927.15 | 927.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 951.15 | 931.95 | 929.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 935.40 | 938.66 | 933.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 935.40 | 938.66 | 933.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 935.40 | 938.66 | 933.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 940.75 | 938.87 | 934.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 942.65 | 938.87 | 934.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 1034.83 | 1004.05 | 982.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1150.50 | 1156.62 | 1157.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1137.55 | 1152.81 | 1155.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1148.00 | 1146.33 | 1150.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1148.00 | 1146.33 | 1150.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1148.00 | 1146.33 | 1150.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1150.50 | 1146.33 | 1150.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1151.55 | 1147.93 | 1150.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1151.55 | 1147.93 | 1150.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1146.65 | 1147.67 | 1149.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:15:00 | 1175.00 | 1147.67 | 1149.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1175.00 | 1153.14 | 1152.18 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 1138.15 | 1149.74 | 1150.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 1131.50 | 1146.09 | 1149.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 1139.25 | 1135.39 | 1141.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 10:15:00 | 1139.25 | 1135.39 | 1141.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1139.25 | 1135.39 | 1141.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 1145.90 | 1135.39 | 1141.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1138.00 | 1135.92 | 1141.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:15:00 | 1149.50 | 1135.92 | 1141.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1151.60 | 1139.05 | 1142.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 1150.90 | 1139.05 | 1142.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1150.00 | 1141.24 | 1142.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:15:00 | 1155.20 | 1141.24 | 1142.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 1155.00 | 1146.19 | 1145.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 1160.25 | 1152.30 | 1148.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1166.20 | 1172.11 | 1164.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1166.20 | 1172.11 | 1164.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1166.20 | 1172.11 | 1164.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 1162.60 | 1172.11 | 1164.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1173.80 | 1172.45 | 1165.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 1165.60 | 1172.45 | 1165.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1174.90 | 1172.16 | 1166.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:45:00 | 1182.50 | 1174.61 | 1168.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 12:30:00 | 693.55 | 2023-05-29 09:15:00 | 706.25 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-05-23 14:30:00 | 695.35 | 2023-05-29 09:15:00 | 706.25 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-05-24 12:15:00 | 695.10 | 2023-05-29 09:15:00 | 706.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-05-25 11:45:00 | 694.40 | 2023-05-29 09:15:00 | 706.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-05-30 11:00:00 | 704.55 | 2023-06-05 09:15:00 | 775.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-26 11:15:00 | 754.00 | 2023-06-27 15:15:00 | 770.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2023-06-26 13:00:00 | 755.00 | 2023-06-27 15:15:00 | 770.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2023-06-26 13:30:00 | 755.25 | 2023-06-27 15:15:00 | 770.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2023-06-26 15:00:00 | 755.05 | 2023-06-27 15:15:00 | 770.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-07-03 09:15:00 | 809.45 | 2023-07-03 11:15:00 | 774.00 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2023-07-03 13:00:00 | 794.40 | 2023-07-06 11:15:00 | 794.80 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2023-07-04 09:15:00 | 799.95 | 2023-07-06 11:15:00 | 794.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-07-06 10:15:00 | 789.55 | 2023-07-06 11:15:00 | 794.80 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2023-07-10 11:45:00 | 775.50 | 2023-07-10 15:15:00 | 794.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2023-07-13 11:15:00 | 798.05 | 2023-07-18 15:15:00 | 797.80 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-07-13 12:30:00 | 798.95 | 2023-07-18 15:15:00 | 797.80 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-07-13 14:45:00 | 798.75 | 2023-07-18 15:15:00 | 797.80 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-14 09:30:00 | 802.80 | 2023-07-18 15:15:00 | 797.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-07-18 09:15:00 | 803.20 | 2023-07-18 15:15:00 | 797.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-07-24 09:15:00 | 825.05 | 2023-08-01 13:15:00 | 848.45 | STOP_HIT | 1.00 | 2.84% |
| SELL | retest2 | 2023-08-04 10:30:00 | 831.45 | 2023-08-07 09:15:00 | 852.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2023-08-04 11:30:00 | 833.00 | 2023-08-07 09:15:00 | 852.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2023-08-04 13:45:00 | 830.80 | 2023-08-07 09:15:00 | 852.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2023-08-04 14:45:00 | 833.15 | 2023-08-07 09:15:00 | 852.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-09-05 09:15:00 | 855.10 | 2023-09-08 11:15:00 | 857.80 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2023-09-12 11:30:00 | 847.80 | 2023-09-18 09:15:00 | 863.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-09-12 15:15:00 | 847.10 | 2023-09-18 09:15:00 | 863.40 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2023-09-13 09:45:00 | 842.00 | 2023-09-18 09:15:00 | 863.40 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2023-09-13 12:15:00 | 850.25 | 2023-09-18 09:15:00 | 863.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-09-26 10:45:00 | 829.40 | 2023-09-28 11:15:00 | 842.55 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-09-27 09:30:00 | 824.45 | 2023-09-28 11:15:00 | 842.55 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2023-10-09 10:15:00 | 845.85 | 2023-10-17 09:15:00 | 930.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-09 11:30:00 | 845.90 | 2023-10-17 09:15:00 | 930.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-09 13:00:00 | 845.75 | 2023-10-17 09:15:00 | 930.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-09 14:00:00 | 845.40 | 2023-10-17 09:15:00 | 929.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-10 09:15:00 | 850.10 | 2023-10-17 09:15:00 | 935.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-12 13:30:00 | 847.00 | 2023-10-17 09:15:00 | 931.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-15 10:45:00 | 960.95 | 2023-11-17 13:15:00 | 939.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-11-15 11:15:00 | 960.05 | 2023-11-17 13:15:00 | 939.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-11-21 13:00:00 | 936.55 | 2023-11-23 09:15:00 | 889.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-22 09:15:00 | 936.50 | 2023-11-23 09:15:00 | 889.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-21 13:00:00 | 936.55 | 2023-11-24 10:15:00 | 890.90 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2023-11-22 09:15:00 | 936.50 | 2023-11-24 10:15:00 | 890.90 | STOP_HIT | 0.50 | 4.87% |
| BUY | retest2 | 2023-12-18 12:15:00 | 1024.70 | 2023-12-19 09:15:00 | 1021.25 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-12-18 13:45:00 | 1025.00 | 2023-12-19 11:15:00 | 1006.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-12-18 14:15:00 | 1024.80 | 2023-12-19 11:15:00 | 1006.80 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2023-12-18 14:45:00 | 1024.40 | 2023-12-19 11:15:00 | 1006.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2023-12-19 09:15:00 | 1032.40 | 2023-12-19 11:15:00 | 1006.80 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2023-12-20 11:15:00 | 998.85 | 2023-12-28 13:15:00 | 948.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 11:15:00 | 998.85 | 2024-01-01 09:15:00 | 939.40 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2024-01-09 11:30:00 | 936.05 | 2024-01-11 11:15:00 | 943.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-01-10 14:00:00 | 934.05 | 2024-01-11 11:15:00 | 943.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-01-31 11:15:00 | 983.60 | 2024-01-31 11:15:00 | 984.85 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-02-19 11:00:00 | 893.10 | 2024-02-21 09:15:00 | 926.40 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2024-02-19 11:45:00 | 894.35 | 2024-02-21 09:15:00 | 926.40 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-02-19 14:30:00 | 893.95 | 2024-02-21 09:15:00 | 926.40 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-03-11 09:15:00 | 855.00 | 2024-03-15 11:15:00 | 812.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 855.00 | 2024-03-15 12:15:00 | 823.80 | STOP_HIT | 0.50 | 3.65% |
| BUY | retest2 | 2024-03-20 09:15:00 | 857.90 | 2024-04-02 14:15:00 | 943.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-16 10:15:00 | 884.75 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-04-18 10:00:00 | 882.10 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-04-18 11:30:00 | 884.80 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2024-04-18 12:15:00 | 884.65 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-04-22 10:45:00 | 879.25 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-04-24 14:15:00 | 875.00 | 2024-04-24 14:15:00 | 879.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-04-30 11:15:00 | 905.70 | 2024-05-02 09:15:00 | 885.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-05-03 09:15:00 | 897.30 | 2024-05-07 10:15:00 | 885.40 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-03 13:45:00 | 897.65 | 2024-05-07 10:15:00 | 885.40 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-06 09:15:00 | 903.35 | 2024-05-07 10:15:00 | 885.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-05-09 09:15:00 | 883.00 | 2024-05-10 09:15:00 | 838.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 883.00 | 2024-05-13 12:15:00 | 794.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-16 15:15:00 | 833.00 | 2024-05-21 12:15:00 | 819.35 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-05-24 10:45:00 | 813.45 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-05-27 10:15:00 | 815.05 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-05-27 11:30:00 | 814.30 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-05-27 15:15:00 | 814.95 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-05-28 10:30:00 | 811.00 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-05-29 10:00:00 | 812.00 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-05-29 11:15:00 | 813.15 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-05-29 12:45:00 | 813.20 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-05-30 09:15:00 | 802.20 | 2024-05-31 13:15:00 | 816.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-08 11:45:00 | 1049.95 | 2024-07-10 09:15:00 | 1094.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2024-07-08 12:45:00 | 1051.95 | 2024-07-10 09:15:00 | 1094.00 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2024-07-10 10:30:00 | 1054.35 | 2024-07-11 10:15:00 | 1075.25 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-07-10 13:00:00 | 1051.05 | 2024-07-11 10:15:00 | 1075.25 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-07-18 09:15:00 | 1092.10 | 2024-07-19 15:15:00 | 1074.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-07-19 09:45:00 | 1090.45 | 2024-07-19 15:15:00 | 1074.05 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-19 12:00:00 | 1083.35 | 2024-07-19 15:15:00 | 1074.05 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-08-01 09:15:00 | 1034.85 | 2024-08-05 13:15:00 | 983.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:45:00 | 1035.15 | 2024-08-05 13:15:00 | 983.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:45:00 | 1035.00 | 2024-08-05 13:15:00 | 983.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 13:15:00 | 1033.90 | 2024-08-05 13:15:00 | 982.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 09:15:00 | 1034.85 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 0.50 | -0.16% |
| SELL | retest2 | 2024-08-01 11:45:00 | 1035.15 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-08-01 12:45:00 | 1035.00 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2024-08-01 13:15:00 | 1033.90 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 0.50 | -0.25% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1007.65 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-08-05 10:30:00 | 1023.00 | 2024-08-06 09:15:00 | 1036.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-06 09:30:00 | 1018.95 | 2024-08-06 14:15:00 | 1040.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1025.80 | 2024-08-06 14:15:00 | 1040.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-06 12:00:00 | 1021.00 | 2024-08-06 14:15:00 | 1040.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-08-16 12:30:00 | 1007.50 | 2024-08-19 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-16 13:30:00 | 1010.45 | 2024-08-19 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-08-16 14:15:00 | 1010.00 | 2024-08-19 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-08-23 09:15:00 | 1052.05 | 2024-08-30 09:15:00 | 1157.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 11:45:00 | 1051.25 | 2024-08-30 09:15:00 | 1156.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-11 12:15:00 | 1100.45 | 2024-09-12 10:15:00 | 1138.45 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-09-11 12:45:00 | 1100.00 | 2024-09-12 10:15:00 | 1138.45 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-09-11 13:30:00 | 1100.70 | 2024-09-12 10:15:00 | 1138.45 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-09-25 15:15:00 | 1305.00 | 2024-09-26 10:15:00 | 1286.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-10-09 09:15:00 | 1339.00 | 2024-10-10 12:15:00 | 1244.95 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2024-10-24 11:00:00 | 1093.35 | 2024-10-25 09:15:00 | 1205.85 | STOP_HIT | 1.00 | -10.29% |
| SELL | retest2 | 2024-11-05 14:45:00 | 1147.80 | 2024-11-07 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-11-06 10:15:00 | 1147.30 | 2024-11-07 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-11-14 14:15:00 | 1042.50 | 2024-11-19 09:15:00 | 1073.05 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-11-14 15:15:00 | 1041.80 | 2024-11-19 09:15:00 | 1073.05 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-11-18 10:15:00 | 1045.00 | 2024-11-19 09:15:00 | 1073.05 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-11-18 11:00:00 | 1043.90 | 2024-11-19 09:15:00 | 1073.05 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-11-26 13:45:00 | 1113.15 | 2024-11-27 09:15:00 | 1091.55 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-11-28 15:00:00 | 1091.40 | 2024-12-02 09:15:00 | 982.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-29 09:15:00 | 1087.40 | 2024-12-02 09:15:00 | 978.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-12 10:00:00 | 1027.30 | 2024-12-12 11:15:00 | 1036.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-12 12:30:00 | 1027.30 | 2024-12-12 14:15:00 | 1040.15 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-12-16 12:00:00 | 1042.70 | 2024-12-17 12:15:00 | 1028.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-17 11:45:00 | 1040.05 | 2024-12-17 12:15:00 | 1028.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-12-20 10:30:00 | 1017.65 | 2024-12-27 10:15:00 | 1024.90 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-01-02 09:45:00 | 1060.70 | 2025-01-06 12:15:00 | 1038.40 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-01-06 12:00:00 | 1060.55 | 2025-01-06 12:15:00 | 1038.40 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-01-09 10:45:00 | 1050.00 | 2025-01-09 11:15:00 | 1058.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1044.55 | 2025-01-21 09:15:00 | 1025.95 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-01-20 09:15:00 | 1044.20 | 2025-01-21 09:15:00 | 1025.95 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-01-24 14:30:00 | 992.90 | 2025-01-28 10:15:00 | 943.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 992.90 | 2025-01-29 10:15:00 | 962.70 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-01-27 09:15:00 | 971.30 | 2025-01-31 12:15:00 | 984.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2025-02-12 09:15:00 | 948.70 | 2025-02-13 11:15:00 | 961.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-14 09:15:00 | 950.50 | 2025-02-19 10:15:00 | 902.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 950.50 | 2025-02-19 13:15:00 | 923.55 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2025-03-07 10:15:00 | 1007.10 | 2025-03-11 09:15:00 | 997.45 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-03-10 09:15:00 | 1004.90 | 2025-03-11 09:15:00 | 997.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-03-10 09:45:00 | 1004.15 | 2025-03-11 09:15:00 | 997.45 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-03-10 11:30:00 | 1006.30 | 2025-03-11 09:15:00 | 997.45 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-03-27 12:15:00 | 1034.40 | 2025-03-27 14:15:00 | 1048.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-04-08 11:15:00 | 987.45 | 2025-04-09 09:15:00 | 1030.10 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-04-08 15:00:00 | 984.75 | 2025-04-09 09:15:00 | 1030.10 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1165.50 | 2025-05-12 13:15:00 | 1165.40 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-05-12 10:45:00 | 1165.90 | 2025-05-12 13:15:00 | 1165.40 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-05-12 11:15:00 | 1165.60 | 2025-05-12 13:15:00 | 1165.40 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-05-12 12:00:00 | 1164.60 | 2025-05-12 13:15:00 | 1165.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-05-28 11:15:00 | 1165.40 | 2025-05-30 13:15:00 | 1281.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-11 13:15:00 | 1270.90 | 2025-06-16 09:15:00 | 1271.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-06-12 10:30:00 | 1273.30 | 2025-06-16 09:15:00 | 1271.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-06-17 13:30:00 | 1276.90 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-17 14:00:00 | 1276.00 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-17 14:45:00 | 1280.00 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-18 11:45:00 | 1277.70 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-19 09:15:00 | 1293.80 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-19 09:45:00 | 1288.50 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-06-19 11:30:00 | 1291.80 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-06-19 13:15:00 | 1290.00 | 2025-06-20 12:15:00 | 1264.80 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1342.10 | 2025-06-27 14:15:00 | 1476.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-02 12:15:00 | 1349.00 | 2025-07-03 13:15:00 | 1361.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-02 15:15:00 | 1348.00 | 2025-07-03 13:15:00 | 1361.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-03 10:00:00 | 1347.00 | 2025-07-03 13:15:00 | 1361.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-11 10:15:00 | 1398.10 | 2025-07-17 13:15:00 | 1397.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-07-11 11:45:00 | 1396.50 | 2025-07-17 13:15:00 | 1397.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-07-14 09:15:00 | 1400.00 | 2025-07-17 13:15:00 | 1397.70 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-07-14 14:30:00 | 1398.00 | 2025-07-17 13:15:00 | 1397.70 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-07-15 09:15:00 | 1404.40 | 2025-07-17 13:15:00 | 1397.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-19 13:00:00 | 1278.90 | 2025-08-28 09:15:00 | 1265.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-21 10:30:00 | 1268.00 | 2025-08-28 09:15:00 | 1265.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-08-21 12:30:00 | 1270.30 | 2025-08-28 09:15:00 | 1265.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-09-05 09:15:00 | 1258.40 | 2025-09-09 14:15:00 | 1254.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-09-16 11:30:00 | 1307.90 | 2025-09-17 09:15:00 | 1280.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-09-18 14:15:00 | 1279.20 | 2025-09-24 11:15:00 | 1277.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-09-18 15:00:00 | 1277.80 | 2025-09-24 11:15:00 | 1277.50 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-09-19 09:15:00 | 1268.90 | 2025-09-24 11:15:00 | 1277.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-29 14:45:00 | 1193.90 | 2025-10-01 14:15:00 | 1235.40 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1203.30 | 2025-10-01 14:15:00 | 1235.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-09-30 13:45:00 | 1205.90 | 2025-10-01 14:15:00 | 1235.40 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-10-01 10:45:00 | 1202.00 | 2025-10-01 14:15:00 | 1235.40 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-10-09 11:00:00 | 1210.20 | 2025-10-10 13:15:00 | 1219.90 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-10-09 12:15:00 | 1210.50 | 2025-10-10 13:15:00 | 1219.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-10 09:30:00 | 1211.00 | 2025-10-10 13:15:00 | 1219.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1228.00 | 2025-10-17 10:15:00 | 1229.40 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-10-24 12:00:00 | 1223.00 | 2025-10-31 10:15:00 | 1212.20 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-10-24 13:00:00 | 1221.80 | 2025-10-31 10:15:00 | 1212.20 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-27 10:30:00 | 1220.60 | 2025-10-31 10:15:00 | 1212.20 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-11-17 14:00:00 | 1215.80 | 2025-11-19 13:15:00 | 1190.90 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1178.30 | 2025-11-24 15:15:00 | 1119.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1178.30 | 2025-11-26 09:15:00 | 1143.20 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-12-12 14:15:00 | 1174.20 | 2025-12-17 09:15:00 | 1166.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-12 14:45:00 | 1175.20 | 2025-12-17 09:15:00 | 1166.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1134.20 | 2026-01-02 10:15:00 | 1118.90 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-12-22 10:30:00 | 1141.50 | 2026-01-02 10:15:00 | 1118.90 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-01-08 10:30:00 | 1043.20 | 2026-01-12 14:15:00 | 1061.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-03 11:15:00 | 1160.00 | 2026-02-03 13:15:00 | 1163.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2026-02-10 09:15:00 | 1187.30 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-12 09:45:00 | 1177.90 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-02-12 10:15:00 | 1177.60 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-02-13 09:30:00 | 1178.30 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1186.90 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1184.60 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-02-16 09:45:00 | 1188.90 | 2026-02-16 11:15:00 | 1171.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-03-17 11:15:00 | 994.50 | 2026-03-20 11:15:00 | 944.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 13:30:00 | 996.90 | 2026-03-20 11:15:00 | 947.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 994.50 | 2026-03-23 11:15:00 | 897.21 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-03-17 13:30:00 | 996.90 | 2026-03-23 12:15:00 | 895.05 | TARGET_HIT | 0.50 | 10.22% |
| BUY | retest2 | 2026-04-02 11:30:00 | 940.75 | 2026-04-08 09:15:00 | 1034.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:15:00 | 942.65 | 2026-04-08 09:15:00 | 1036.91 | TARGET_HIT | 1.00 | 10.00% |
